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
    required = case.get("required_candidates", [])
    valid_first = case.get("valid_first_tools", [])
    candidates = [name for name in headers.get("x-mcp-prism-tool-names", "").split(",") if name]
    first = selected[0] if selected else None
    no_tool = not required
    return {"id": case["id"], "workload": case["workload"], "mode": mode,
            "required_candidates": required, "valid_first_tools": valid_first,
            "candidate_names": candidates, "selected": selected,
            "candidate_recall": set(required).issubset(candidates),
            "first_step_correct": (first in valid_first) if valid_first else (first is None),
            "no_tool_correct": (first is None) if no_tool else None,
            "wrong_first_tool": first is not None and first not in valid_first,
            "candidate_tools": int(headers.get("x-mcp-prism-tools", 0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    parser.add_argument("--data", type=Path, default=Path("data/private/benchmark_requests.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    rows = [invoke(args.url, case, mode) for case in cases for mode in ("baseline", "prism")]
    summary = {}
    for mode in ("baseline", "prism"):
        chosen = [row for row in rows if row["mode"] == mode]
        tool_cases = [row for row in chosen if row["required_candidates"]]
        no_tool_cases = [row for row in chosen if not row["required_candidates"]]
        summary[mode] = {
            "n": len(chosen),
            "candidate_recall": sum(row["candidate_recall"] for row in tool_cases) / len(tool_cases),
            "first_step_accuracy": sum(row["first_step_correct"] for row in chosen) / len(chosen),
            "wrong_first_tool_rate": sum(row["wrong_first_tool"] for row in chosen) / len(chosen),
            "no_tool_accuracy": sum(row["no_tool_correct"] for row in no_tool_cases) / len(no_tool_cases),
            "mean_candidate_tools": sum(row["candidate_tools"] for row in chosen) / len(chosen),
        }
    baseline, prism = summary["baseline"], summary["prism"]
    gates = {
        "candidate_recall_95": prism["candidate_recall"] >= 0.95,
        "no_tool_95": prism["no_tool_accuracy"] >= 0.95,
        "first_step_noninferior": prism["first_step_accuracy"] >= baseline["first_step_accuracy"],
        "wrong_tool_noninferior": prism["wrong_first_tool_rate"] <= baseline["wrong_first_tool_rate"],
    }
    args.out.write_text(json.dumps({"raw": rows, "summary": summary, "gates": gates}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gates": gates}, indent=2))
    if not all(gates.values()):
        raise SystemExit("quality gate failed")


if __name__ == "__main__":
    main()

