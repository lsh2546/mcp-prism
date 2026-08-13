from mcp_prism.canonical import canonical_tool_bundle
from mcp_prism.models import ToolDefinition


def tool(name):
    return ToolDefinition("server", name, "  Example   tool ", {"properties": {"b": {}, "a": {}}, "type": "object"})


def test_bundle_order_and_fingerprint_are_stable():
    first = canonical_tool_bundle([tool("z"), tool("a")])
    second = canonical_tool_bundle([tool("a"), tool("z")])
    assert first == second

