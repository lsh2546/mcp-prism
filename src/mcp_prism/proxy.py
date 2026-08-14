Exit code: 0
Wall time: 0.3 seconds
Output:
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .canonical import canonical_tool_bundle, openai_tool
from .encoder import EncoderConfig, Int8OnnxEncoder
from .hierarchical import HierarchicalRouter
from .index import SemanticToolIndex
from .models import ToolDefinition


def load_tools(path: Path) -> list[ToolDefinition]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [ToolDefinition(row["server"], row["name"], row["description"], row["input_schema"]) for row in rows]


class ProxyEngine:
    def __init__(self, tools: list[ToolDefinition], router: HierarchicalRouter, upstream: str):
        self.tools = tuple(tools)
        self.router = router
        self.upstream = upstream.rstrip("/")

    def transform(self, request: dict[str, Any], mode: str, cache_prompt: bool) -> tuple[dict[str, Any], dict[str, str]]:
        payload = dict(request)
        messages = payload.get("messages", [])
        query = next(
            (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), ""
        )
        if mode == "baseline":
            tools = list(self.tools)
            fingerprint, encoded = canonical_tool_bundle(tools)
            tasks = 1
        elif mode == "prism":
            decision = self.router.route(str(query))
            tools = [item.tool for item in decision.candidates]
            fingerprint, encoded = canonical_tool_bundle(tools)
            tasks = len(decision.tasks)
        else:
            raise ValueError("x-mcp-prism-mode must be baseline or prism")
        payload["tools"] = encoded
        payload["cache_prompt"] = cache_prompt
        # Stable metadata is intentionally not inserted into messages; doing so would perturb the measured prefix.
        return payload, {
            "x-mcp-prism-mode": mode,
            "x-mcp-prism-tools": str(len(tools)),
            "x-mcp-prism-prefix": fingerprint,
            "x-mcp-prism-atomic-tasks": str(tasks),
            "x-mcp-prism-schema-chars": str(len(json.dumps(encoded, separators=(",", ":")))),
            "x-mcp-prism-tool-names": ",".join(tool.qualified_name for tool in tools),
        }


class PrismHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: ProxyEngine

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/health"}:
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
            mode = self.headers.get("x-mcp-prism-mode", "prism")
            cache_prompt = self.headers.get("x-mcp-prism-cache", "on").lower() == "on"
            payload, prism_headers = self.engine.transform(request, mode, cache_prompt)
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            upstream = urllib.request.Request(
                f"{self.engine.upstream}/v1/chat/completions",
                data=body,
                headers={"content-type": "application/json", "authorization": self.headers.get("authorization", "")},
                method="POST",
            )
            with urllib.request.urlopen(upstream, timeout=600) as response:
                self.send_response(response.status)
                content_type = response.headers.get("content-type", "application/json")
                self.send_header("content-type", content_type)
                for key, value in prism_headers.items():
                    self.send_header(key, value)
                if request.get("stream"):
                    self.send_header("connection", "close")
                    self.end_headers()
                    while True:
                        chunk = response.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    self.close_connection = True
                else:
                    result = response.read()
                    self.send_header("content-length", str(len(result)))
                    self.end_headers()
                    self.wfile.write(result)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))
        except urllib.error.HTTPError as error:
            result = error.read()
            self.send_response(error.code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)
        except Exception as error:
            self.send_error(502, str(error))

    def log_message(self, format: str, *args: object) -> None:
        return


def build_engine(root: Path, upstream: str) -> ProxyEngine:
    tools = load_tools(root / "data/private/tool_catalog.json")
    encoder = Int8OnnxEncoder(
        EncoderConfig(root / "models/encoder-qint8-arm64.onnx", root / "models/tokenizer.json")
    )
    router = HierarchicalRouter(SemanticToolIndex.build(tools, encoder))
    return ProxyEngine(tools, router, upstream)


def serve(host: str = "127.0.0.1", port: int = 8090, upstream: str = "http://127.0.0.1:8080") -> None:
    root = Path(__file__).resolve().parents[2]
    PrismHandler.engine = build_engine(root, upstream)
    ThreadingHTTPServer((host, port), PrismHandler).serve_forever()

