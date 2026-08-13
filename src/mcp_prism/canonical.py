from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .models import ToolDefinition


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.qualified_name,
            "description": " ".join(tool.description.split()),
            "parameters": tool.input_schema,
        },
    }


def canonical_tool_bundle(tools: Iterable[ToolDefinition]) -> tuple[str, list[dict[str, Any]]]:
    payload = [openai_tool(tool) for tool in sorted(tools, key=lambda item: item.qualified_name)]
    encoded = canonical_json(payload)
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return fingerprint, payload

