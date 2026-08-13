from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    source: str
    query: str
    required_tools: frozenset[str]
    expected_arguments: dict[str, dict[str, Any]]
    no_tool: bool = False
    category: str = "single_tool"


@dataclass(frozen=True)
class CasePrediction:
    retrieved_tools: tuple[str, ...]
    called_tools: tuple[str, ...]
    arguments: dict[str, dict[str, Any]]
    task_succeeded: bool


def normalized(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in sorted(value.items())}
    return value


def score_cases(cases: Iterable[EvaluationCase], predictions: Iterable[CasePrediction]) -> dict[str, float]:
    pairs = list(zip(cases, predictions, strict=True))
    if not pairs:
        raise ValueError("at least one evaluation case is required")
    recall_hits = final_hits = argument_hits = successes = wrong_calls = refusals = no_tool_count = 0
    argument_total = 0
    for case, prediction in pairs:
        required = set(case.required_tools)
        retrieved = set(prediction.retrieved_tools)
        called = set(prediction.called_tools)
        if required.issubset(retrieved):
            recall_hits += 1
        if called == required:
            final_hits += 1
        wrong_calls += int(bool(called - required))
        successes += int(prediction.task_succeeded)
        if case.no_tool:
            no_tool_count += 1
            refusals += int(not called)
        for tool, expected in case.expected_arguments.items():
            argument_total += 1
            argument_hits += int(normalized(prediction.arguments.get(tool)) == normalized(expected))
    count = len(pairs)
    return {
        "recall_at_k": recall_hits / count,
        "final_tool_accuracy": final_hits / count,
        "argument_accuracy": argument_hits / argument_total if argument_total else 1.0,
        "task_success_rate": successes / count,
        "wrong_tool_call_rate": wrong_calls / count,
        "no_tool_refusal_accuracy": refusals / no_tool_count if no_tool_count else 1.0,
    }


def quality_gate(baseline: dict[str, float], prism: dict[str, float]) -> tuple[bool, tuple[str, ...]]:
    higher = ("recall_at_k", "final_tool_accuracy", "argument_accuracy", "task_success_rate", "no_tool_refusal_accuracy")
    lower = ("wrong_tool_call_rate",)
    failures = [f"{metric} regressed" for metric in higher if prism[metric] < baseline[metric]]
    failures.extend(f"{metric} increased" for metric in lower if prism[metric] > baseline[metric])
    return not failures, tuple(failures)

