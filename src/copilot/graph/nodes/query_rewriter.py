"""Node 4: rewrite the query when private evidence is insufficient (bounded retries)."""
from __future__ import annotations

from copilot.generation.evidence import neutralize_delimiters
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.graph.schemas import QueryRewrite
from copilot.graph.state import CopilotState
from copilot.llm.client import LLMUnavailableError, StructuredOutputError
from copilot.llm.prompts import REWRITER_SYSTEM_PROMPT

_MAX_QUERY_CHARS = 300


def make_query_rewriter(deps: GraphDeps):
    """Build the ``query_rewriter`` node."""
    s = deps.settings

    @traced("query_rewriter")
    def query_rewriter(state: CopilotState) -> dict:
        grade = state.get("private_grade", {})
        previous = [state["question"], *state.get("rewritten_queries", [])]
        missing = grade.get("missing_information") or []
        prompt = (
            f"<question>\n{neutralize_delimiters(state['question'])}\n</question>\n"
            f"Previous queries: {neutralize_delimiters(' | '.join(previous))}\n"
            f"Missing information reported by the grader: {neutralize_delimiters('; '.join(missing) or 'unknown')}"
        )
        usage = None
        try:
            rewrite, usage = deps.llm.structured(
                [{"role": "system", "content": REWRITER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], QueryRewrite
            )
        except (LLMUnavailableError, StructuredOutputError) as exc:
            return _stop(s.max_rewrite_retries, f"rewriter failed: {type(exc).__name__}", None)

        new_query = " ".join(rewrite.rewritten_query.split())[:_MAX_QUERY_CHARS]
        if not new_query or new_query.lower() in {p.lower() for p in previous}:
            return _stop(s.max_rewrite_retries, "rewriter produced no new query", usage)
        retry = state.get("retry_count", 0) + 1
        return {
            "current_query": new_query,
            "rewritten_queries": [new_query],
            "retry_count": retry,
            "llm_usage": usage_update(usage),
            "_event": event(
                f"retry {retry}/{s.max_rewrite_retries}: rewrote query", "retry private retrieval",
                rewritten_query=new_query, missing_concepts=rewrite.missing_concepts, added_terms=rewrite.added_terms,
                retry_count=retry,
            ),
        }

    return query_rewriter


def _stop(max_retries: int, why: str, usage) -> dict:
    """Stop retrying: mark the rewrite as failed so routing moves on (no loop)."""
    return {
        "rewrite_failed": True,
        "retry_count": max_retries,
        "llm_usage": usage_update(usage),
        "_event": event(why, "stop retrying; fall through to external search or abstention"),
    }
