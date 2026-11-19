"""Second-stage reranker: re-scores retrieved chunks with a local cross-encoder.

The bi-encoder (MiniLM) used for Pinecone search embeds the question and each chunk separately, which is fast
but coarse. A cross-encoder reads the question and one chunk together and scores how well the chunk answers it.
It only reorders chunks the access filter already allowed; it never adds or reveals documents. If the model
cannot be loaded or scoring fails, the original vector order is kept (retrieval must not break because of it).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from copilot.config import Settings, get_settings
from copilot.logging_config import get_logger
from copilot.vectorstore.models import RetrievedChunk

log = get_logger(__name__)


class CrossEncoderReranker:
    """Lazy wrapper around ``sentence_transformers.CrossEncoder``."""

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        self._model_name = model_name
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy: heavy import

            log.info("Loading reranker model", extra={"model": self._model_name})
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], bool]:
        """Return ``(chunks ordered by cross-encoder score, reranked)``.

        ``reranked`` is False when the input is returned unchanged (nothing to rank, or the model failed).
        """
        if len(chunks) < 2:
            return list(chunks), False
        try:
            scores = self._get_model().predict([(query, c.text) for c in chunks])
            scored = [c.model_copy(update={"rerank_score": float(s)}) for c, s in zip(chunks, scores)]
        except Exception as exc:  # noqa: BLE001 - never let the optional reranker break retrieval
            log.warning("Reranking failed; keeping vector order", extra={"error": type(exc).__name__})
            return list(chunks), False
        scored.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else float("-inf"), reverse=True)
        return scored, True


@lru_cache(maxsize=2)
def _cached(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    """Process-wide reranker for the configured model (the model loads once, on first use)."""
    return _cached((settings or get_settings()).rerank_model_name)
