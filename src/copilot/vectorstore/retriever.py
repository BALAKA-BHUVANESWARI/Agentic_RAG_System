"""Private retriever: embeds the query and fetches only chunks the role may read.

Used by the LangGraph ``private_retriever`` node and by the baseline RAG, so the
retrieval logic exists in exactly one place.
"""
from __future__ import annotations

from typing import Any

from copilot.auth.access import allowed_access_levels, resolve_role
from copilot.config import Settings, get_settings
from copilot.logging_config import get_logger
from copilot.vectorstore.models import RetrievalFilters, RetrievalResult, RetrievedChunk

log = get_logger(__name__)


def limit_per_document(chunks: list[RetrievedChunk], cap: int, k: int) -> list[RetrievedChunk]:
    """Pick up to ``k`` chunks in rank order with at most ``cap`` per document, then refill unused slots.

    A SOFT cap: when too few other documents are available, the best chunks over the cap fill the remaining
    slots, so a single-document answer is never cut short. The returned list keeps the original rank order.
    """
    selected: list[int] = []
    over_cap: list[int] = []
    counts: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        if len(selected) >= k:
            break
        if counts.get(chunk.document_id, 0) < cap:
            selected.append(i)
            counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
        else:
            over_cap.append(i)
    if len(selected) < k:
        selected += over_cap[: k - len(selected)]
    return [chunks[i] for i in sorted(selected)]


class Retriever:
    """Access-filtered semantic retrieval over the Pinecone index."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Any | None = None,
        store: Any | None = None,
        reranker: Any | None = None,
    ):
        self._settings = settings or get_settings()
        if embedder is None:
            from copilot.embeddings.service import get_embedding_service

            embedder = get_embedding_service()
        if store is None:
            from copilot.vectorstore.pinecone_store import PineconeStore

            store = PineconeStore(self._settings)
        if reranker is None and self._settings.rerank_enabled:
            from copilot.vectorstore.reranker import get_reranker

            reranker = get_reranker(self._settings)
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        role: str | None = None,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        """Retrieve authorised chunks for ``query``.

        Access control is enforced twice: by a Pinecone metadata filter and by a
        post-filter that drops any returned chunk the role may not read.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        s = self._settings
        role = resolve_role(role, s)
        k = top_k or s.retrieval_top_k
        threshold = s.retrieval_min_score if min_score is None else min_score
        levels = allowed_access_levels(role, s)
        if not levels:
            return RetrievalResult(query=query, role=role, top_k=k, status="no_access")

        filters = filters or RetrievalFilters()
        vector = self._embedder.embed_query(query)
        cap = s.retrieval_max_chunks_per_doc
        fetch_k = max(k, s.rerank_candidates) if (self._reranker is not None or cap) else k
        raw = self._store.query(
            vector,
            fetch_k,
            levels,
            category=filters.category,
            department=filters.department,
            document_ids=filters.document_ids,
        )
        authorized = [c for c in raw if c.access_level in levels]
        dropped_unauthorized = len(raw) - len(authorized)
        if dropped_unauthorized:
            log.warning("Dropped unauthorized chunks from retrieval", extra={"count": dropped_unauthorized})
        kept = [c for c in authorized if c.score >= threshold]
        dropped_low_score = len(authorized) - len(kept)
        kept.sort(key=lambda c: c.score, reverse=True)
        reranked = False
        if self._reranker is not None and len(kept) > 1:
            kept, reranked = self._reranker.rerank(query, kept)
        if cap:
            kept = limit_per_document(kept, cap, k)
        elif self._reranker is not None:
            kept = kept[:k]
        return RetrievalResult(
            query=query,
            role=role,
            top_k=k,
            status="ok" if kept else "empty",
            chunks=kept,
            dropped_unauthorized=dropped_unauthorized,
            dropped_low_score=dropped_low_score,
            reranked=reranked,
        )
