"""Private retriever: embeds the query and fetches only chunks the role may read.

Used by the LangGraph ``private_retriever`` node and by the baseline RAG, so the
retrieval logic exists in exactly one place.
"""
from __future__ import annotations

from typing import Any

from copilot.auth.access import allowed_access_levels, resolve_role
from copilot.config import Settings, get_settings
from copilot.logging_config import get_logger
from copilot.vectorstore.models import RetrievalFilters, RetrievalResult

log = get_logger(__name__)


class Retriever:
    """Access-filtered semantic retrieval over the Pinecone index."""

    def __init__(self, settings: Settings | None = None, embedder: Any | None = None, store: Any | None = None):
        self._settings = settings or get_settings()
        if embedder is None:
            from copilot.embeddings.service import get_embedding_service

            embedder = get_embedding_service()
        if store is None:
            from copilot.vectorstore.pinecone_store import PineconeStore

            store = PineconeStore(self._settings)
        self._embedder = embedder
        self._store = store

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
        raw = self._store.query(
            vector,
            k,
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
        kept.sort(key=lambda c: c.score, reverse=True)
        return RetrievalResult(
            query=query,
            role=role,
            top_k=k,
            status="ok" if kept else "empty",
            chunks=kept,
            dropped_unauthorized=dropped_unauthorized,
            dropped_low_score=len(authorized) - len(kept),
        )
