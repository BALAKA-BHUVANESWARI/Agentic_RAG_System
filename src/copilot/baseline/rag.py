"""Baseline RAG: private retrieval -> LLM generation. No grading, rewriting, web search or validation.

Serves as the comparison point for the agentic workflow (Phase 7) and exercises the
shared retriever and answer generator end to end.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel

from copilot.config import Settings, get_settings
from copilot.generation.answer import AnswerGenerator, make_abstention
from copilot.generation.evidence import from_retrieved_chunks
from copilot.generation.models import GeneratedAnswer
from copilot.logging_config import bind_request_id, get_logger, redact
from copilot.vectorstore.models import RetrievalResult

log = get_logger(__name__)


class RAGResponse(BaseModel):
    """Result of one baseline RAG request."""

    request_id: str
    mode: Literal["baseline"] = "baseline"
    question: str
    role: str
    answer: GeneratedAnswer
    retrieval: RetrievalResult | None = None
    latency_ms: float
    error: str | None = None


class BaselineRAG:
    """retrieve -> generate. Dependencies are injectable for testing."""

    def __init__(
        self, settings: Settings | None = None, retriever: Any | None = None, generator: AnswerGenerator | None = None
    ) -> None:
        self._settings = settings or get_settings()
        if retriever is None:
            from copilot.vectorstore.retriever import Retriever

            retriever = Retriever(self._settings)
        self._retriever = retriever
        self._generator = generator or AnswerGenerator(self._settings)

    def answer(self, question: str, role: str | None = None, request_id: str | None = None) -> RAGResponse:
        """Answer ``question`` for ``role``. Service failures become abstentions, not exceptions."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        started = time.perf_counter()
        with bind_request_id(request_id) as rid:
            retrieval: RetrievalResult | None = None
            error: str | None = None
            try:
                retrieval = self._retriever.retrieve(question, role)
            except Exception as exc:  # noqa: BLE001 - Pinecone/embedding outage
                error = redact(f"{type(exc).__name__}: {exc}")[:300]
                log.error("Retrieval failed", extra={"error": error})
                answer = make_abstention("retrieval_unavailable", error=error)
            else:
                answer = self._generator.generate(question, from_retrieved_chunks(retrieval.chunks))
            log.info(
                "Baseline answered",
                extra={"abstained": answer.abstained, "reason": answer.abstention_reason,
                       "chunks": len(retrieval.chunks) if retrieval else 0},
            )
            return RAGResponse(
                request_id=rid, question=question, role=retrieval.role if retrieval else (role or self._settings.default_role),
                answer=answer, retrieval=retrieval, latency_ms=(time.perf_counter() - started) * 1000, error=error,
            )
