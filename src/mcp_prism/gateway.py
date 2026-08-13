from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical import canonical_tool_bundle
from .models import GatewayResult
from .router import AdaptiveRouter


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
    raise ValueError("request has no user message")


class PrismGateway:
    def __init__(self, router: AdaptiveRouter):
        self.router = router

    def transform(self, request: dict[str, Any]) -> GatewayResult:
        messages = request.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        query = last_user_text(messages)
        decision = self.router.route(query)
        fingerprint, tools = canonical_tool_bundle(item.tool for item in decision.selected)
        upstream = deepcopy(request)
        upstream["tools"] = tools
        upstream["cache_prompt"] = True
        upstream.setdefault("metadata", {})["mcp_prism_prefix"] = fingerprint
        return GatewayResult(
            upstream_request=upstream,
            routing=decision,
            headers={
                "x-mcp-prism-tools": str(decision.candidate_count),
                "x-mcp-prism-prefix": fingerprint,
                "x-mcp-prism-schema-reduction": f"{decision.schema_reduction_pct:.2f}",
            },
        )
