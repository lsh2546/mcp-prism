from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .encoder import Int8OnnxEncoder
from .models import RankedTool, ToolDefinition


def schema_fields(schema: dict) -> tuple[str, str, str]:
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    names = " ".join(sorted(properties))
    descriptions = " ".join(
        str(value.get("description", "")) for _, value in sorted(properties.items()) if isinstance(value, dict)
    )
    required = " ".join(sorted(schema.get("required", []))) if isinstance(schema, dict) else ""
    return names, descriptions, required


def retrieval_document(tool: ToolDefinition) -> str:
    names, descriptions, required = schema_fields(tool.input_schema)
    return "\n".join(
        [
            f"tool identity: {tool.server} {tool.name.replace('_', ' ')}",
            f"capability: {tool.description}",
            f"input names: {names}",
            f"input meanings: {descriptions}",
            f"required inputs: {required}",
        ]
    )


@dataclass
class SemanticToolIndex:
    tools: tuple[ToolDefinition, ...]
    matrix: np.ndarray
    encoder: Int8OnnxEncoder

    @classmethod
    def build(cls, tools: Iterable[ToolDefinition], encoder: Int8OnnxEncoder) -> "SemanticToolIndex":
        ordered = tuple(sorted(tools, key=lambda item: item.qualified_name))
        if not ordered:
            raise ValueError("at least one tool is required")
        matrix = encoder.encode(retrieval_document(tool) for tool in ordered)
        return cls(ordered, matrix, encoder)

    def rank(self, query: str) -> list[RankedTool]:
        query_vector = self.encoder.encode([query])[0]
        scores = self.matrix @ query_vector
        indices = np.argsort(-scores, kind="stable")
        return [RankedTool(self.tools[int(index)], float(scores[index])) for index in indices]

    def rank_many(self, queries: Iterable[str]) -> list[RankedTool]:
        """Max-pool semantic evidence across decomposed request intents."""
        values = [value.strip() for value in queries if value.strip()]
        vectors = self.encoder.encode(values)
        scores = vectors @ self.matrix.T
        pooled = scores.max(axis=0)
        indices = np.argsort(-pooled, kind="stable")
        return [RankedTool(self.tools[int(index)], float(pooled[index])) for index in indices]
