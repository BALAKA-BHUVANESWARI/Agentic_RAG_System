"""Node 5: public web search via Tavily, only for sanitised, non-sensitive queries."""
from __future__ import annotations

from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced
from copilot.graph.state import CopilotState
from copilot.search.sanitizer import sanitize_query
from copilot.search.tavily_client import ExternalSearchError


def make_external_search(deps: GraphDeps):
    """Build the ``external_search`` node. Routing decides IF this runs; it re-checks safety."""
    s = deps.settings

    @traced("external_search")
    def external_search(state: CopilotState) -> dict:
        cls = state.get("classification", {})
        blocked = None
        if not s.external_search_enabled:
            blocked = "external search is disabled by configuration"
        elif not state.get("external_search_allowed", True):
            blocked = "external search is not allowed for this request"
        elif cls.get("contains_sensitive"):
            blocked = f"question flagged as sensitive ({cls.get('sensitive_reason') or 'no reason given'})"
        if blocked:
            return _blocked(blocked)

        base = cls.get("public_search_query") or state["question"]
        sanitized = sanitize_query(base, s.private_terms, s.max_external_query_chars)
        if sanitized.blocked:
            return _blocked(sanitized.blocked_reason or "blocked by sanitizer")
        if not sanitized.usable:
            return _blocked("no safe, generic public query could be formed after removing internal details")

        calls_before = getattr(deps.search, "call_count", 0)
        try:
            results = deps.search.search(sanitized.query)
        except ExternalSearchError as exc:
            calls = max(getattr(deps.search, "call_count", 0) - calls_before, 1)
            return {
                "external_triggered": True, "external_query": sanitized.query, "external_results": [],
                "external_blocked_reason": f"search service unavailable: {exc}", "tavily_calls": calls,
                "_event": event("Tavily search failed", "continue: no external evidence", query=sanitized.query, error=str(exc)),
            }
        calls = max(getattr(deps.search, "call_count", 0) - calls_before, 1)
        return {
            "external_triggered": True,
            "external_query": sanitized.query,
            "external_results": [r.model_dump() for r in results],
            "tavily_calls": calls,
            "_event": event(
                f"{len(results)} public result(s) for sanitised query", "continue: grade external evidence",
                query=sanitized.query, removed_from_query=sanitized.removed,
                results=[f"{r.domain}{' (official)' if r.preferred else ''}: {r.title}" for r in results],
            ),
        }

    return external_search


def _blocked(reason: str) -> dict:
    return {
        "external_triggered": False, "external_results": [], "external_blocked_reason": reason,
        "_event": event(f"external search not performed: {reason}", "continue without external evidence", reason=reason),
    }
