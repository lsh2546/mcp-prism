
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def request_once(url: str, row: dict[str, Any], mode: str, cache: bool) -> dict[str, Any]:
    payload = {
        "model": row.get("model", "mcp-prism"),
        "messages": [{"role": "system", "content": "Select the minimum tools needed. Never invent arguments."},
                     {"role": "user", "content": row["query"]}],
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        # Performance path measures prefill and TTFT only. Full tool-call generations
        # are evaluated separately so decode time cannot obscure the prefill effect.
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "content-type": "application/json",
        "x-mcp-prism-mode": mode,
        "x-mcp-prism-cache": "on" if cache else "off",
    })
    started = time.perf_counter()
    ttft = None
    usage: dict[str, Any] = {}
    tool_names: list[str] = []
    with urllib.request.urlopen(req, timeout=600) as response:
        headers = dict(response.headers.items())
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            choices = event.get("choices") or []
            if ttft is None and choices:
                delta = choices[0].get("delta", {})
                if delta.get("content") or delta.get("tool_calls"):
                    ttft = time.perf_counter() - started
            for call in (choices[0].get("delta", {}).get("tool_calls", []) if choices else []):
                name = call.get("function", {}).get("name")
                if name and name not in tool_names:
                    tool_names.append(name)
            if event.get("usage"):
                usage = event["usage"]
    latency = time.perf_counter() - started
    return {
        "id": row["id"], "workload": row["workload"], "mode": mode, "cache_prompt": cache,
        "latency_s": latency, "ttft_s": ttft or latency, "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"), "selected_tools": tool_names,
        "candidate_tools": int(headers.get("x-mcp-prism-tools", 0)),
        "prefix": headers.get("x-mcp-prism-prefix"),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["workload"], row["mode"], row["cache_prompt"]), []).append(row)
    out = []
    for (workload, mode, cache), items in sorted(groups.items()):
        lat = [x["latency_s"] for x in items]
        ttft = [x["ttft_s"] for x in items]
        prompts = [x["prompt_tokens"] for x in items if x["prompt_tokens"] is not None]
        out.append({"workload": workload, "mode": mode, "cache_prompt": cache, "n": len(items),
                    "p50_latency_s": percentile(lat, .5), "p95_latency_s": percentile(lat, .95),
                    "p50_ttft_s": percentile(ttft, .5), "p95_ttft_s": percentile(ttft, .95),
                    "requests_per_s": len(items) / sum(lat),
                    "mean_prompt_tokens": statistics.mean(prompts) if prompts else None})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    parser.add_argument("--data", type=Path, default=Path("data/private/benchmark_requests.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    # One fixed representative per workload, repeated independently, gives a
    # meaningful p95 without mixing workload classes or paying decode cost.
    representatives = {}
    for case in cases:
        representatives.setdefault(case["workload"], case)
    cases = list(representatives.values())
    rows: list[dict[str, Any]] = []
    cache_modes = (True,) if args.cache_only else (False, True)
    schedule = [(mode, cache) for cache in cache_modes for mode in ("baseline", "prism")]
    for i in range(args.warmup + args.repetitions):
        for case_index, case in enumerate(cases):
            ordered = schedule if (i + case_index) % 2 == 0 else list(reversed(schedule))
            for mode, cache in ordered:
                result = request_once(args.url, case, mode, cache)
                if i >= args.warmup:
                    result["iteration"] = i - args.warmup
                    rows.append(result)
    metadata = {"architecture": platform.machine(), "platform": platform.platform(),
                "processor": platform.processor(), "cpu_count": os.cpu_count(),
                "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    result = {"metadata": metadata, "raw": rows, "summary": summarize(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

