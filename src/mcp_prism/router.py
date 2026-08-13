from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
import json

import numpy as np

from .canonical import canonical_json, canonical_tool_bundle, openai_tool
from .models import RankedTool, RoutingDecision, ToolDefinition
from .index import SemanticToolIndex


@dataclass(frozen=True)
class RouterConfig:
    initial_k: int = 3
    expanded_k: int = 8
    min_top_score: float = 0.32
    min_margin: float = 0.04
    no_tool_threshold: float = 0.18


def decompose_intents(query: str) -> tuple[str, ...]:
    parts = re.split(r"(?:\s+and\s+|\s+then\s+|,\s*(?:and\s+|then\s+)?|;)", query, flags=re.IGNORECASE)
    values = tuple(part.strip(" .") for part in parts if len(part.strip(" .")) >= 4)
    return values or (query,)


class AdaptiveRouter:
    def __init__(
        self,
        index: SemanticToolIndex,
        config: RouterConfig | None = None,
        calibration_path: Path = Path("data/private/calibration_requests.json"),
    ):
        self.index = index
        self.tools = index.tools
        self.config = config or RouterConfig()
        rows = json.loads(calibration_path.read_text(encoding="utf-8"))
        vectors = self.index.encoder.encode(row["query"] for row in rows)
        self.tool_centroid = vectors[[row["needs_tool"] for row in rows]].mean(axis=0)
        self.no_tool_centroid = vectors[[not row["needs_tool"] for row in rows]].mean(axis=0)
        self.tool_centroid /= np.linalg.norm(self.tool_centroid)
        self.no_tool_centroid /= np.linalg.norm(self.no_tool_centroid)

    def needs_tool(self, query: str) -> tuple[bool, float]:
        vector = self.index.encoder.encode([query])[0]
        tool_score = float(self.tool_centroid @ vector)
        no_tool_score = float(self.no_tool_centroid @ vector)
        return tool_score >= no_tool_score, tool_score - no_tool_score

    def route(self, query: str) -> RoutingDecision:
        intents = decompose_intents(query)
        per_intent = [self.index.rank(intent) for intent in intents]
        score_by_name = {}
        tool_by_name = {}
        guaranteed = []
        for intent_ranking in per_intent:
            for item in intent_ranking[: self.config.initial_k]:
                guaranteed.append(item.tool.qualified_name)
            for item in intent_ranking:
                name = item.tool.qualified_name
                tool_by_name[name] = item.tool
                score_by_name[name] = max(score_by_name.get(name, -1.0), item.score)
        ranked = sorted(
            (RankedTool(tool_by_name[name], score) for name, score in score_by_name.items()),
            key=lambda item: (-item.score, item.tool.qualified_name),
        )
        top = ranked[0].score
        runner_up = ranked[1].score if len(ranked) > 1 else 0.0
        margin = top - runner_up
        tool_needed, tool_margin = self.needs_tool(query)
        no_tool = not tool_needed
        uncertain = not no_tool and (top < self.config.min_top_score or margin < self.config.min_margin)
        target = self.config.expanded_k if uncertain else self.config.initial_k
        selected_names = [] if no_tool else list(dict.fromkeys(guaranteed))
        if not no_tool:
            for item in ranked:
                if len(selected_names) >= target:
                    break
                if item.tool.qualified_name not in selected_names:
                    selected_names.append(item.tool.qualified_name)
        selected = tuple(RankedTool(tool_by_name[name], score_by_name[name]) for name in selected_names)
        count = len(selected)
        fingerprint, prism_tools = canonical_tool_bundle(item.tool for item in selected)
        baseline_tools = [openai_tool(tool) for tool in self.tools]
        confidence = 0.0 if top <= 0 else max(0.0, min(1.0, margin / top))
        return RoutingDecision(
            query=query,
            selected=selected,
            candidate_count=count,
            confidence=confidence,
            expanded=uncertain,
            fingerprint=fingerprint,
            baseline_schema_chars=len(canonical_json(baseline_tools)),
            prism_schema_chars=len(canonical_json(prism_tools)),
        )
