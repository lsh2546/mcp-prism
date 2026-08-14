Exit code: 0
Wall time: 0.4 seconds
Output:
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_prism.proxy import build_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/private/benchmark_requests.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    engine = build_engine(Path.cwd(), "http://127.0.0.1:8080")
    rows = []
    for case in cases:
        candidates = [item.tool.qualified_name for item in engine.router.route(case["query"]).candidates]
        required = case["required_candidates"]
        rows.append({
            "id": case["id"], "workload": case["workload"], "required": required,
            "candidates": candidates, "complete": set(required).issubset(candidates),
            "wrong": sorted(set(candidates) - set(required)),
        })
    tool_rows = [row for row in rows if row["required"]]
    no_tool_rows = [row for row in rows if not row["required"]]
    summary = {
        "candidate_recall": sum(row["complete"] for row in tool_rows) / len(tool_rows),
        "no_tool_accuracy": sum(not row["candidates"] for row in no_tool_rows) / len(no_tool_rows),
        "mean_candidate_tools": sum(len(row["candidates"]) for row in rows) / len(rows),
    }
    gates = {"candidate_recall_95": summary["candidate_recall"] >= .95,
             "no_tool_95": summary["no_tool_accuracy"] >= .95}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "summary": summary, "gates": gates}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "gates": gates}, indent=2))
    if not all(gates.values()):
        raise SystemExit("routing quality gate failed")


if __name__ == "__main__":
    main()

