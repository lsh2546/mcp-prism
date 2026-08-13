from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def invoke(url: str, case: dict, mode: str) -> dict:
    payload = {
        "model": "mcp-prism",
        "messages": [
            {"role": "system", "content": "Select the minimum tools needed. Never invent arguments."},
            {"role": "user", "content": case["query"]},
        ],
        "temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 96, "stream": False,
    }
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
        "content-type": "application/json", "x-mcp-prism-mode": mode, "x-mcp-prism-cache": "off",
    })
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
        headers = dict(response.headers.items())
    message = result.get("choices", [{}])[0].get("message", {})
    calls = message.get("tool_calls") or []
    selected = [call.get("function", {}).get("name") for call in calls if call.get("function", {}).get("name")]
    expected = case.get("expected", [])
    return {"id": case["id"], "workload": case["workload"], "mode": mode, "expected": expected,
            "selected": selected, "complete": set(expected).issubset(selected),
            "wrong": sorted(set(selected) - set(expected)), "candidate_tools": int(headers.get("x-mcp-prism-tools", 0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    parser.add_argument("--data", type=Path, default=Path("data/private/benchmark_requests.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))[:6]
    rows = [invoke(args.url, case, mode) for case in cases for mode in ("baseline", "prism")]
    summary = {}
    for mode in ("baseline", "prism"):
        chosen = [row for row in rows if row["mode"] == mode]
        summary[mode] = {"n": len(chosen), "complete_rate": sum(row["complete"] for row in chosen) / len(chosen),
                         "wrong_tool_rate": sum(bool(row["wrong"]) for row in chosen) / len(chosen)}
    args.out.write_text(json.dumps({"raw": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
