"""Node 2: retrieve authorised internal chunks for the current (original or rewritten) query."""
from __future__ import annotations

from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced
from copilot.graph.state import CopilotState
from copilot.logging_config import redact


def make_private_retriever(deps: GraphDeps):
    """Build the ``private_retriever`` node."""

    @traced("private_retriever")
    def private_retriever(state: CopilotState) -> dict:
        query = state["current_query"]
        attempt = state.get("retry_count", 0)
        try:
            result = deps.retriever.retrieve(query, state.get("role"))
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
            "attempt": attempt, "query": query, "status": result.status,
            "chunk_ids": [c.chunk_id for c in chunks],
            "document_ids": sorted({c.document_id for c in chunks}),
            "scores": [{"chunk_id": c.chunk_id, "document": c.document_name, "score": round(c.score, 4)} for c in chunks],
            "dropped_unauthorized": result.dropped_unauthorized,
        }
        return {
            "retrieved_chunks": [c.model_dump() for c in chunks],
            "retrieval_attempts": [record],
            "_event": event(
                f"attempt {attempt}: {len(chunks)} chunk(s) retrieved (status={result.status})",
                "continue: grade evidence" if chunks else "continue: nothing retrieved",
                attempt=attempt, query=query, role=result.role,
                documents=[f"{c.document_name} ({c.score:.2f})" for c in chunks],
                dropped_unauthorized=result.dropped_unauthorized,
            ),
        }

    return private_retriever
