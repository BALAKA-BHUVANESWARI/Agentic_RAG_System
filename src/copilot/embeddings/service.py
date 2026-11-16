"""Local sentence-embedding service (HuggingFace all-MiniLM-L6-v2, 384-dim)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from copilot.config import Settings, get_settings
from copilot.logging_config import get_logger

log = get_logger(__name__)


class EmbeddingService:
    """Wraps a SentenceTransformer model; loads it lazily on first use.

    Vectors are L2-normalised so Pinecone's cosine metric behaves as expected.
    The first use downloads the model from HuggingFace (needs network once).
    """

    def __init__(self, model_name: str, dimension: int, batch_size: int, model: Any | None = None) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._model = model
        self._checked = False

    @property
    def dimension(self) -> int:
        """Expected embedding dimension."""
        return self._dimension

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            log.info("Loading embedding model", extra={"model": self._model_name})
            self._model = SentenceTransformer(self._model_name)
        if not self._checked:
            # sentence-transformers renamed this method; support both versions.
            get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            actual = get_dim()
            if actual != self._dimension:
                raise RuntimeError(
                    f"Embedding model produces {actual}-dim vectors but EMBEDDING_DIMENSION={self._dimension}."
                )
            self._checked = True
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts (batched)."""
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_documents([text])[0]


def build_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Create an :class:`EmbeddingService` from settings."""
    settings = settings or get_settings()
    return EmbeddingService(
        settings.embedding_model_name, settings.embedding_dimension, settings.embedding_batch_size
    )


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Process-wide cached embedding service (avoids reloading the model)."""
    return build_embedding_service()
