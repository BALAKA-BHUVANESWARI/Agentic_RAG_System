"""Node 2: retrieve authorised internal chunks for the current (original or rewritten) query."""
from __future__ import annotations

from copilot.graph.deps import GraphDeps
from copilot.graph.expansion import expand_query
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.graph.state import CopilotState
from copilot.logging_config import get_logger, redact
from copilot.vectorstore.retriever import merge_retrievals

log = get_logger(__name__)


def make_private_retriever(deps: GraphDeps):
    """Build the ``private_retriever`` node."""

    s = deps.settings

    @traced("private_retriever")
    def private_retriever(state: CopilotState) -> dict:
        query = state["current_query"]
        attempt = state.get("retry_count", 0)
        expanded: list[str] = []
        usage = None
        try:
            result = deps.retriever.retrieve(query, state.get("role"))
            # EXPERIMENT C: only before the first retrieval of the request (never on a rewrite retry), and only when the flag is on.
            if s.retrieval_query_expansion and attempt == 0 and not state.get("rewritten_queries") and result.status != "no_access":
                expanded, usage = expand_query(deps.llm, state["question"], s.retrieval_query_expansion)
                extra = []
                for paraphrase in expanded:
                    try:
                        extra.append(deps.retriever.retrieve(paraphrase, state.get("role")))
                    except Exception as exc:  # noqa: BLE001 - keep the original result if an extra lookup fails
                        log.warning("Expanded-query retrieval failed; ignoring it", extra={"error": type(exc).__name__})
                if extra:
                    result = merge_retrievals([result, *extra], s.retrieval_top_k)
        except Exception as exc:  # noqa: BLE001 - Pinecone/embedding outage
            error = redact(f"{type(exc).__name__}: {exc}")[:300]
            return {
                "retrieved_chunks": [],
                "force_abstain": {"reason": "retrieval_unavailable", "error": error},
                "retrieval_attempts": [{"attempt": attempt, "query": query, "status": "error", "chunk_ids": [], "document_ids": [], "scores": [], "error": error}],
                "_event": event(f"retrieval failed: {type(exc).__name__}", "abstain: knowledge base unavailable", attempt=attempt, error=error),
            }
        chunks = result.chunks
        record = {
            "expanded_queries": expanded,
            "attempt": attempt, "query": query, "status": result.status,
            "chunk_ids": [c.chunk_id for c in chunks],
            "document_ids": sorted({c.document_id for c in chunks}),
            "scores": [{"chunk_id": c.chunk_id, "document": c.document_name, "score": round(c.score, 4)} for c in chunks],
            "dropped_unauthorized": result.dropped_unauthorized,
            "reranked": result.reranked,
        }
        update = {
            "retrieved_chunks": [c.model_dump() for c in chunks],
            "retrieval_attempts": [record],
            "_event": event(
                f"attempt {attempt}: {len(chunks)} chunk(s) retrieved (status={result.status})",
                "continue: grade evidence" if chunks else "continue: nothing retrieved",
                attempt=attempt, query=query, role=result.role,
                documents=[f"{c.document_name} ({c.score:.2f})" for c in chunks],
                dropped_unauthorized=result.dropped_unauthorized,
                reranked=result.reranked,
                expanded_queries=expanded,
            ),
        }
        if expanded:
            update["expanded_queries"] = expanded
        if usage:
            update["llm_usage"] = usage_update(usage)
        return update

    return private_retriever
