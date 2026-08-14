
from mcp_prism.hierarchical import execution_frontier
from mcp_prism.models import RankedTool, ToolDefinition


def test_execution_frontier_hides_dependent_workflow_tools():
    schema = {"type": "object", "properties": {}}
    candidates = tuple(
        RankedTool(ToolDefinition(server, name, name, schema), 1.0)
        for server, name in (
            ("files", "search_files"),
            ("files", "share_file"),
            ("gmail", "send_message"),
        )
    )
    frontier = execution_frontier("Find the PDF, share it, and email it", candidates)
    assert [item.tool.qualified_name for item in frontier] == ["files.search_files"]


def test_execution_frontier_advances_dependency_tiers():
    schema = {"type": "object", "properties": {}}
    candidates = tuple(
        RankedTool(ToolDefinition(server, name, name, schema), 1.0)
        for server, name in (
            ("github", "list_workflow_runs"),
            ("github", "get_workflow_logs"),
            ("slack", "post_message"),
        )
    )
    after_search = execution_frontier(
        "Find a failed run, read logs, and post the cause",
        candidates,
        ("github.list_workflow_runs",),
    )
    assert [item.tool.qualified_name for item in after_search] == ["github.get_workflow_logs"]

