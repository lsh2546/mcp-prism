# Private MCP scenarios

`tool_catalog.json` is a hand-authored catalog of overlapping tools modeled after common Gmail, Calendar, GitHub, Slack, database, search, file, weather, maps, task, contacts, commerce, and monitoring MCP integrations.

`eval_requests.json` is the separately labeled private evaluation set. Each row records its source as `private_mcp_scenarios_v1`, a traffic category, and the complete set of tools that must be retrieved. Ambiguous cases intentionally accept a broader required set for safety evaluation.

These files are not derived from ToolBench or API-Bank and must never be reported as public benchmark results.

