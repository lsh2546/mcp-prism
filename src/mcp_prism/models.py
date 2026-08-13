from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"{self.server}.{self.name}"


@dataclass(frozen=True)
class RankedTool:
    tool: ToolDefinition
    score: float


@dataclass(frozen=True)
class RoutingDecision:
    query: str
    selected: tuple[RankedTool, ...]
    candidate_count: int
    confidence: float
    expanded: bool
    fingerprint: str
    baseline_schema_chars: int
    prism_schema_chars: int

    @property
    def schema_reduction_pct(self) -> float:
        if self.baseline_schema_chars == 0:
            return 0.0
        return 100.0 * (self.baseline_schema_chars - self.prism_schema_chars) / self.baseline_schema_chars


@dataclass(frozen=True)
class GatewayResult:
    upstream_request: dict[str, Any]
    routing: RoutingDecision
    headers: dict[str, str] = field(default_factory=dict)

