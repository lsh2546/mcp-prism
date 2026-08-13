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
    "slack": {"slack", "channel", "thread"},
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
    "github.get_workflow_logs": {"ci", "workflow", "log", "logs", "failure"},
    "gmail.search_messages": {"email", "inbox", "newest", "latest", "from"},
    "files.search_files": {"file", "pdf", "document", "storage", "find", "search"},
    "calendar.search_events": {"meeting", "changed", "check", "find"},
}


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
        values = list(dict.fromkeys(explicit + [self.domain_names[int(i)] for i in order[: self.domain_k]]))
        return tuple(values)

    def hybrid_score(self, task: str, item: RankedTool) -> float:
        query_words = words(task)
        tool_words = words(f"{item.tool.name} {item.tool.description} {item.tool.input_schema}")
        overlap = len(query_words & tool_words) / max(1, len(query_words))
        action = tool_action(item.tool)
        action_match = any(alias in query_words for alias in ACTION_ALIASES.get(action, set()))
        service_match = bool(query_words & SERVICE_CUES.get(item.tool.server, {item.tool.server.lower()}))
        tool_match = bool(query_words & TOOL_CUES.get(item.tool.qualified_name, set()))
        return item.score + 0.20 * overlap + 0.18 * action_match + 0.30 * service_match + 0.24 * tool_match

    def route(self, query: str) -> HierarchicalDecision:
        pieces = decompose_intents(query)
        tasks = tuple(AtomicTask(piece, self.infer_domains(piece)) for piece in pieces)
        selected: dict[str, RankedTool] = {}
        for task in tasks:
            allowed_servers = set().union(*(DOMAIN_SERVERS[domain] for domain in task.domains))
            ranked = [item for item in self.index.rank(task.text) if item.tool.server in allowed_servers]
            reranked = sorted(ranked, key=lambda item: (-self.hybrid_score(task.text, item), item.tool.qualified_name))
            for item in reranked[: self.tools_per_task]:
                score = self.hybrid_score(task.text, item)
                previous = selected.get(item.tool.qualified_name)
                if previous is None or score > previous.score:
                    selected[item.tool.qualified_name] = RankedTool(item.tool, score)
        candidates = tuple(sorted(selected.values(), key=lambda item: (-item.score, item.tool.qualified_name)))
        # Multiple equally plausible communication/search domains require clarification rather than fake certainty.
        clarification = len(tasks) == 1 and len(tasks[0].domains) > 1 and max((item.score for item in candidates), default=0) < 0.45
        return HierarchicalDecision(tasks, candidates, clarification)
