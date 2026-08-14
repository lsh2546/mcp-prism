
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .index import SemanticToolIndex
from .models import RankedTool, ToolDefinition
from .router import decompose_intents


DOMAIN_SERVERS = {
    "communication": {"gmail", "slack", "contacts"},
    "source_control": {"github"},
    "calendar": {"calendar"},
    "files": {"files"},
    "database": {"database"},
    "web": {"search", "weather", "maps"},
    "project_management": {"tasks"},
    "commerce": {"commerce"},
    "operations": {"monitoring"},
}

DOMAIN_CARDS = {
    "communication": "email inbox message reply send Slack channel conversation contact person address",
    "source_control": "GitHub repository source code issue pull request CI workflow run build log branch",
    "calendar": "calendar meeting event attendee schedule free time appointment",
    "files": "private connected storage file folder document PDF path share move rename",
    "database": "SQL database table schema query transaction row",
    "web": "public web news academic image weather forecast map route place current information",
    "project_management": "project task assignee status due date work item",
    "commerce": "customer order inventory product warehouse refund purchase",
    "operations": "production service logs metrics CPU memory errors incident alert monitoring",
}

ACTION_ALIASES = {
    "search": {"find", "search", "look", "locate", "check"},
    "read": {"read", "show", "open", "inspect", "details", "log"},
    "create": {"create", "schedule", "open", "add", "draft"},
    "send": {"send", "post", "email", "reply", "share", "invite"},
    "update": {"update", "change", "move", "rename", "modify"},
    "delete": {"delete", "remove", "cancel", "refund"},
}

SERVICE_CUES = {
    "gmail": {"email", "inbox", "mail"},
    "slack": {"slack", "channel", "thread", "discussion", "conversation"},
    "github": {"github", "repository", "repo", "ci", "workflow", "pull", "issue", "branch"},
    "calendar": {"calendar", "meeting", "appointment", "attendee"},
    "files": {"file", "folder", "document", "pdf", "path", "storage"},
    "database": {"database", "sql", "table", "query"},
    "search": {"web", "news", "paper", "academic", "image"},
    "commerce": {"order", "inventory", "sku", "refund", "customer"},
    "monitoring": {"production", "service", "metric", "cpu", "memory", "error", "incident", "logs"},
    "tasks": {"task", "project", "assignee", "due"},
}

TOOL_CUES = {
    "github.list_workflow_runs": {"ci", "workflow", "failed", "runs"},
    "github.search_issues": {"issue", "issues", "relevant", "attach", "find", "search"},
    "github.get_workflow_logs": {"ci", "workflow", "log", "logs", "failure"},
    "gmail.search_messages": {"email", "inbox", "newest", "latest", "from"},
    "files.search_files": {"file", "pdf", "document", "storage", "find", "search"},
    "calendar.search_events": {"meeting", "changed", "check", "find"},
    "weather.daily_forecast": {"tomorrow", "forecast", "daily", "week", "weekend"},
    "weather.current_conditions": {"current", "currently", "now", "conditions"},
    "files.share_file": {"share", "link", "access", "permission"},
    "gmail.send_message": {"email", "send", "mail"},
    "slack.post_message": {"post", "send", "slack", "channel"},
    "slack.search_messages": {"search", "find", "latest", "discussion", "conversation"},
}

NO_TOOL_PATTERNS = (
    "without calling any external tool",
    "without using any tool",
    "do not call any tool",
    "no external tool",
)


def words(text: str) -> set[str]:
    return {value.lower() for value in re.findall(r"[A-Za-z0-9_]+", text)}


def tool_action(tool: ToolDefinition) -> str:
    name = tool.name.lower()
    for action in ACTION_ALIASES:
        if name.startswith(action) or action in name:
            return action
    if any(value in name for value in ("post", "reply", "share", "upload")):
        return "send"
    if any(value in name for value in ("move", "modify")):
        return "update"
    return "other"


@dataclass(frozen=True)
class AtomicTask:
    text: str
    domains: tuple[str, ...]


@dataclass(frozen=True)
class HierarchicalDecision:
    tasks: tuple[AtomicTask, ...]
    candidates: tuple[RankedTool, ...]
    clarification_required: bool


class HierarchicalRouter:
    def __init__(self, index: SemanticToolIndex, domain_k: int = 2, tools_per_task: int = 3):
        self.index = index
        self.domain_k = domain_k
        self.tools_per_task = tools_per_task
        self.domain_names = tuple(DOMAIN_CARDS)
        self.domain_vectors = index.encoder.encode(DOMAIN_CARDS[name] for name in self.domain_names)

    def infer_domains(self, text: str) -> tuple[str, ...]:
        vector = self.index.encoder.encode([text])[0]
        scores = self.domain_vectors @ vector
        order = np.argsort(-scores, kind="stable")
        explicit = [
            domain
            for domain, servers in DOMAIN_SERVERS.items()
            if any(server.lower() in words(text) for server in servers)
        ]
        lowered = text.lower()
        if any(cue in lowered for cue in ("meeting", "design review", "appointment", "calendar event")):
            explicit.insert(0, "calendar")
        if any(cue in lowered for cue in ("discussion", "conversation", "slack channel", "thread")):
            explicit.insert(0, "communication")
        if any(cue in lowered for cue in ("tomorrow's weather", "weather tomorrow", "forecast")):
            explicit.insert(0, "web")
        values = list(dict.fromkeys(explicit + [self.domain_names[int(i)] for i in order[: self.domain_k]]))
        return tuple(values)

    def hybrid_score(self, task: str, item: RankedTool) -> float:
        query_words = words(task)
        tool_words = words(f"{item.tool.name} {item.tool.description} {item.tool.input_schema}")
        overlap = len(query_words & tool_words) / max(1, len(query_words))
        action = tool_action(item.tool)
        action_match = any(alias in query_words for alias in ACTION_ALIASES.get(action, set()))
        service_match = bool(query_words & SERVICE_CUES.get(item.tool.server, {item.tool.server.lower()}))
        cue_words = TOOL_CUES.get(item.tool.qualified_name, set())
        cue_match = len(query_words & cue_words) / max(1, min(2, len(cue_words)))
        return item.score + 0.20 * overlap + 0.18 * action_match + 0.30 * service_match + 0.36 * cue_match

    def route(self, query: str) -> HierarchicalDecision:
        if any(pattern in query.lower() for pattern in NO_TOOL_PATTERNS):
            return HierarchicalDecision((AtomicTask(query, tuple()),), tuple(), False)
        pieces = decompose_intents(query)
        tasks = tuple(AtomicTask(piece, self.infer_domains(piece)) for piece in pieces)
        selected: dict[str, RankedTool] = {}
        for task in tasks:
            allowed_servers = set().union(*(DOMAIN_SERVERS[domain] for domain in task.domains))
            ranked = [item for item in self.index.rank(task.text) if item.tool.server in allowed_servers]
            reranked = sorted(ranked, key=lambda item: (-self.hybrid_score(task.text, item), item.tool.qualified_name))
            # Preserve at least one candidate from every inferred domain. A single
            # global Top-K lets a strong source-control score erase a necessary
            # communication step in cross-service workflows.
            for domain in task.domains:
                domain_items = [item for item in reranked if item.tool.server in DOMAIN_SERVERS[domain]]
                if domain_items:
                    item = domain_items[0]
                    score = self.hybrid_score(task.text, item)
                    previous = selected.get(item.tool.qualified_name)
                    if previous is None or score > previous.score:
                        selected[item.tool.qualified_name] = RankedTool(item.tool, score)
            for item in reranked[: self.tools_per_task]:
                score = self.hybrid_score(task.text, item)
                previous = selected.get(item.tool.qualified_name)
                if previous is None or score > previous.score:
                    selected[item.tool.qualified_name] = RankedTool(item.tool, score)
        candidates = tuple(sorted(selected.values(), key=lambda item: (-item.score, item.tool.qualified_name)))
        # Multiple equally plausible communication/search domains require clarification rather than fake certainty.
        clarification = len(tasks) == 1 and len(tasks[0].domains) > 1 and max((item.score for item in candidates), default=0) < 0.45
        return HierarchicalDecision(tasks, candidates, clarification)

    def workflow_plan(self, query: str, candidates: tuple[RankedTool, ...]) -> tuple[RankedTool, ...]:
        """Choose one executable operation per expressed atomic task, in request order."""
        if not candidates:
            return ()
        plan: list[RankedTool] = []
        used: set[str] = set()
        pieces: list[str] = []
        for piece in decompose_intents(query):
            if piece.lower().startswith("compare ") and re.search(r"\s+with\s+", piece, re.IGNORECASE):
                pieces.extend(value.strip() for value in re.split(r"\s+with\s+", piece, maxsplit=1, flags=re.IGNORECASE))
            else:
                pieces.append(piece)
        for piece in pieces:
            domains = self.infer_domains(piece)
            allowed = set().union(*(DOMAIN_SERVERS[domain] for domain in domains))
            pool = [item for item in candidates if item.tool.server in allowed] or list(candidates)
            ranked = sorted(
                pool,
                key=lambda item: (-self.hybrid_score(piece, item), item.tool.qualified_name),
            )
            choice = next((item for item in ranked if item.tool.qualified_name not in used), None)
            if choice is not None:
                plan.append(choice)
                used.add(choice.tool.qualified_name)
        return tuple(plan)


def execution_frontier(
    query: str,
    candidates: tuple[RankedTool, ...],
    completed: tuple[str, ...] = (),
) -> tuple[RankedTool, ...]:
    """Return tools whose inputs are available before any workflow result exists."""
    lowered = query.lower()
    sequenced = any(token in lowered for token in (" and ", " then ", ",", "referenced", "relevant", " its ", " it "))
    remaining = tuple(item for item in candidates if item.tool.qualified_name not in completed)
    if not sequenced:
        return remaining

    def independent(item: RankedTool) -> bool:
        name = item.tool.name.lower()
        return (
            name.startswith(("search_", "list_", "find_"))
            or name in {"current_conditions", "daily_forecast", "weather_alerts", "get_metrics", "geocode", "nearby_places", "route"}
        )

    # Before any result exists, only source operations have satisfiable inputs.
    # Afterwards expose one dependency tier at a time. This prevents a model
    # from posting/refunding before it has read the content or identifier that
    # supplies the required arguments.
    if not completed:
        frontier = tuple(item for item in remaining if independent(item))
        return frontier or remaining

    def tier(item: RankedTool) -> int:
        name = item.tool.name.lower()
        if independent(item):
            return 0
        if name.startswith(("get_", "read_", "download_")):
            return 1
        if any(token in name for token in ("share", "create_link", "upload")):
            return 2
        if any(token in name for token in ("send", "post", "reply", "refund", "update", "delete", "move")):
            return 3
        return 1

    minimum = min((tier(item) for item in remaining), default=0)
    return tuple(item for item in remaining if tier(item) == minimum)

