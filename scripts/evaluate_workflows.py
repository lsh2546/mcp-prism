
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM = (
    "Execute the user's workflow one tool at a time. Call the next dependency-ready tool immediately. "
    "Use prior tool results to fill arguments. When every requested operation is complete, answer briefly. "
    "If no tool is needed, answer normally. Never invent unavailable identifiers."
)


def valid_arguments(arguments: str, schema: dict[str, Any]) -> bool:
    try:
        value = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    return all(name in value for name in schema.get("required", []))


def tool_result(name: str, step: int) -> str:
    return json.dumps({
        "ok": True,
        "tool": name,
        "step": step,
        "id": f"result-{step + 1}",
        "run_id": 7312,
        "path": "/shared/result.pdf",
        "url": "https://example.invalid/shared/result",
        "summary": "Deterministic evaluation result",
        "content": "Deterministic source content",
    })


def invoke(url: str, messages: list[dict[str, Any]], mode: str) -> tuple[dict[str, Any], dict[str, str]]:
    payload = {
        "model": "mcp-prism", "messages": messages,
        "temperature": 0, "top_p": 1, "seed": 42,
        "max_tokens": 128, "stream": False,
    }
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
        "content-type": "application/json", "x-mcp-prism-mode": mode, "x-mcp-prism-cache": "on",
    })
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response), dict(response.headers.items())


def run_case(url: str, case: dict[str, Any], mode: str, schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = case["workflow"]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": case["query"]},
    ]
    actual: list[str] = []
    arguments_ok: list[bool] = []
    candidate_counts: list[int] = []
    for step in range(max(1, len(expected) + 1)):
        result, headers = invoke(url, messages, mode)
        candidate_counts.append(int(headers.get("x-mcp-prism-tools", 0)))
        message = result.get("choices", [{}])[0].get("message", {})
        calls = message.get("tool_calls") or []
        if not calls:
            break
        call = calls[0]
        function = call.get("function", {})
        name = function.get("name", "")
        actual.append(name)
        arguments_ok.append(name in schemas and valid_arguments(function.get("arguments", ""), schemas[name]))
        messages.append({
            "role": "assistant", "content": message.get("content"),
            "tool_calls": [call],
        })
        messages.append({"role": "tool", "tool_call_id": call.get("id", f"call-{step}"),
                         "content": tool_result(name, step)})
        if len(actual) >= len(expected):
            break
    success = actual == expected and all(arguments_ok)
    return {
        "id": case["id"], "mode": mode, "expected": expected, "actual": actual,
        "arguments_valid": arguments_ok, "workflow_success": success,
        "mean_delivered_tools": sum(candidate_counts) / len(candidate_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    parser.add_argument("--data", type=Path, default=Path("data/private/benchmark_requests.json"))
    parser.add_argument("--catalog", type=Path, default=Path("data/private/tool_catalog.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-out", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    schemas = {f"{row['server']}.{row['name']}": row["input_schema"] for row in catalog}
    rows = [run_case(args.url, case, mode, schemas) for case in cases for mode in ("baseline", "prism")]
    summary = {}
    for mode in ("baseline", "prism"):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "n": len(selected),
            "complete_workflow_success": sum(row["workflow_success"] for row in selected) / len(selected),
            "mean_delivered_tools": sum(row["mean_delivered_tools"] for row in selected) / len(selected),
        }
    gates = {
        "workflow_success_85": summary["prism"]["complete_workflow_success"] >= .85,
        "baseline_noninferior": summary["prism"]["complete_workflow_success"] >= summary["baseline"]["complete_workflow_success"],
    }
    output = {"held_out": args.held_out, "raw": rows, "summary": summary, "gates": gates}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not all(gates.values()):
        raise SystemExit("workflow quality gate failed")


if __name__ == "__main__":
    main()

