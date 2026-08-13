from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from .models import ToolDefinition


@dataclass(frozen=True)
class StdioServer:
    name: str
    command: tuple[str, ...]
    environment: dict[str, str] | None = None


class McpProtocolError(RuntimeError):
    pass


class StdioMcpClient:
    """Minimal synchronous MCP client for initialization and paginated tools/list."""

    def __init__(self, server: StdioServer):
        self.server = server
        self.process: subprocess.Popen[str] | None = None
        self.next_id = 1

    def __enter__(self) -> "StdioMcpClient":
        self.process = subprocess.Popen(
            self.server.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=self.server.environment,
        )
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-prism", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _write(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise McpProtocolError("MCP process is not running")
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if self.process is None or self.process.stdout is None:
            raise McpProtocolError("MCP process has no stdout")
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise McpProtocolError(f"MCP process closed before response: {stderr[-500:]}")
            message = json.loads(line)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise McpProtocolError(str(message["error"]))
            return message.get("result", {})

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def list_tools(self) -> list[ToolDefinition]:
        tools: list[ToolDefinition] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params)
            for item in result.get("tools", []):
                tools.append(
                    ToolDefinition(
                        server=self.server.name,
                        name=item["name"],
                        description=item.get("description", ""),
                        input_schema=item.get("inputSchema", {"type": "object", "properties": {}}),
                    )
                )
            cursor = result.get("nextCursor")
            if not cursor:
                return tools


def collect_tools(servers: list[StdioServer]) -> list[ToolDefinition]:
    collected: list[ToolDefinition] = []
    for server in servers:
        with StdioMcpClient(server) as client:
            collected.extend(client.list_tools())
    names = [tool.qualified_name for tool in collected]
    if len(names) != len(set(names)):
        raise ValueError("duplicate qualified tool names")
    return collected

