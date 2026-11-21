"""Pinecone operations: index creation, upsert, filtered query and deletion."""
from __future__ import annotations

import time
from typing import Any, Iterator

from copilot.config import Settings, get_settings
from copilot.ingestion.models import Chunk
from copilot.logging_config import get_logger
from copilot.retry import retry_call
from copilot.vectorstore.models import RetrievedChunk

log = get_logger(__name__)


def chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Pinecone metadata for a chunk. None values are dropped (Pinecone rejects null)."""
    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "title": chunk.title,
        "category": chunk.category,
        "department": chunk.department,
        "access_level": chunk.access_level,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "uploaded_at": chunk.uploaded_at,
        "version": chunk.version,
        "last_reviewed": chunk.last_reviewed,
        "status": chunk.status,
        "superseded_by": chunk.superseded_by,
        "text": chunk.text,
    }
    return {k: v for k, v in metadata.items() if v is not None}


class PineconeStore:
    """Thin wrapper around a Pinecone serverless index.

    ``client``/``index`` can be injected (tests use an in-memory fake index).
    """

    def __init__(self, settings: Settings | None = None, client: Any | None = None, index: Any | None = None):
        self._settings = settings or get_settings()
        self._client = client
        self._index = index

    # ---- connection -----------------------------------------------------
    def _retry(self, operation: str, fn: Any) -> Any:
        s = self._settings
        return retry_call(fn, operation=operation, max_retries=s.service_max_retries, base_delay_s=s.service_retry_base_delay_s)

    def _get_client(self) -> Any:
        if self._client is None:
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self._settings.require_secret("pinecone"))
        return self._client

    def ensure_index(self) -> None:
        """Create the serverless index if missing; verify dimension and metric if present."""
        s = self._settings
        pc = self._get_client()
        if not pc.has_index(s.pinecone_index_name):
            from pinecone import ServerlessSpec

            log.info("Creating Pinecone index", extra={"index": s.pinecone_index_name})
            pc.create_index(
                name=s.pinecone_index_name,
                dimension=s.embedding_dimension,
                metric=s.pinecone_metric,
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )
            for _ in range(60):
                if pc.describe_index(s.pinecone_index_name).status["ready"]:
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Pinecone index was created but did not become ready in time.")
        else:
            desc = pc.describe_index(s.pinecone_index_name)
            if int(desc.dimension) != s.embedding_dimension:
                raise RuntimeError(
                    f"Pinecone index '{s.pinecone_index_name}' has dimension {desc.dimension}, "
                    f"expected {s.embedding_dimension}. Use a different PINECONE_INDEX_NAME."
                )

    @property
    def index(self) -> Any:
        """The Pinecone index handle (created/verified on first access)."""
        if self._index is None:
            self.ensure_index()
            self._index = self._get_client().Index(self._settings.pinecone_index_name)
        return self._index

    # ---- operations -----------------------------------------------------
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Upsert chunk vectors in batches. Returns the number of vectors written."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        s = self._settings
        items = [
            {"id": c.chunk_id, "values": v, "metadata": chunk_to_metadata(c)} for c, v in zip(chunks, vectors)
        ]
        for start in range(0, len(items), s.pinecone_upsert_batch_size):
            batch = items[start : start + s.pinecone_upsert_batch_size]
            self._retry(
                "pinecone.upsert",
                lambda b=batch: self.index.upsert(vectors=b, namespace=s.pinecone_namespace),
            )
        return len(items)

    def query(
        self,
        vector: list[float],
        top_k: int,
        access_levels: list[str],
        *,
        category: str | None = None,
        department: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Similarity search with a mandatory access-level metadata filter."""
        if not access_levels:
            return []
        conditions: list[dict[str, Any]] = [{"access_level": {"$in": list(access_levels)}}]
        if category:
            conditions.append({"category": {"$eq": category}})
        if department:
            conditions.append({"department": {"$eq": department}})
        if document_ids:
            conditions.append({"document_id": {"$in": list(document_ids)}})
        metadata_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        response = self._retry(
            "pinecone.query",
            lambda: self.index.query(
                vector=vector,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True,
                namespace=self._settings.pinecone_namespace,
            ),
        )
        matches = response["matches"] if isinstance(response, dict) else response.matches
        results: list[RetrievedChunk] = []
        for m in matches:
            get = m.get if isinstance(m, dict) else lambda k, d=None, _m=m: getattr(_m, k, d)
            results.append(RetrievedChunk.from_match(get("id"), get("score", 0.0), get("metadata", {})))
        return results

    def delete_document(self, document_id: str, keep_version: int | None = None) -> int:
        """Delete a document's vectors (all versions, or all except ``keep_version``).

        Serverless indexes cannot delete by metadata filter, so vector IDs are
        listed by prefix (``<document_id>:``) and deleted by ID.
        """
        ns = self._settings.pinecone_namespace
        keep_prefix = f"{document_id}:v{keep_version}:" if keep_version is not None else None
        ids: list[str] = []
        for page in self.index.list(prefix=f"{document_id}:", namespace=ns):
            # Older SDKs yield pages of plain id strings; current SDKs yield pages of ListItem objects with an ``id``.
            ids.extend(i for i in ((x if isinstance(x, str) else getattr(x, "id", None)) for x in page) if i)
        if keep_prefix:
            ids = [i for i in ids if not i.startswith(keep_prefix)]
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            self._retry("pinecone.delete", lambda b=batch: self.index.delete(ids=b, namespace=ns))
        return len(ids)

    def iter_chunks(self, batch_size: int = 100) -> Iterator[RetrievedChunk]:
        """Every chunk in this store's namespace with its stored text and metadata (``score`` is 0.0).

        For offline experiments that need the whole corpus (docs/HYBRID_RETRIEVAL.md). It is not used on the request path
        and it is NOT access-filtered: the caller must apply the role's access levels itself.
        """
        ns = self._settings.pinecone_namespace
        ids: list[str] = []
        for page in self.index.list(namespace=ns):
            ids.extend(i for i in ((x if isinstance(x, str) else getattr(x, "id", None)) for x in page) if i)
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            response = self._retry("pinecone.fetch", lambda b=batch: self.index.fetch(ids=b, namespace=ns))
            vectors = response["vectors"] if isinstance(response, dict) else response.vectors
            for vec_id in batch:
                vec = vectors.get(vec_id)
                if vec is None:
                    continue
                metadata = vec.get("metadata") if isinstance(vec, dict) else getattr(vec, "metadata", None)
                yield RetrievedChunk.from_match(vec_id, 0.0, dict(metadata or {}))

    def vector_count(self) -> int:
        """Number of vectors in this store's namespace."""
        stats = self.index.describe_index_stats()
        namespaces = stats["namespaces"] if isinstance(stats, dict) else stats.namespaces
        ns = namespaces.get(self._settings.pinecone_namespace)
        if ns is None:
            return 0
        return int(ns["vector_count"] if isinstance(ns, dict) else ns.vector_count)

    def health_check(self) -> bool:
        """True if the index can be reached."""
        try:
            self.index.describe_index_stats()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Pinecone health check failed: %s", type(exc).__name__)
            return False
