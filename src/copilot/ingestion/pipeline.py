"""Ingestion pipeline: validate -> load -> clean -> metadata -> chunk -> embed -> index.

A reusable library (no Streamlit/FastAPI dependency). Used by the API, scripts and tests.

Design notes:
* New vectors are written BEFORE old-version vectors are deleted, so a failed
  re-index never leaves a document with no vectors.
* Failures that happen before indexing starts leave an already-indexed document
  untouched; the error is recorded in the ingestion history only.
* The raw source file is kept in ``uploads_dir`` so a document can be re-indexed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from copilot.config import Settings, get_settings
from copilot.db.repository import DocumentRecord, DocumentRepository
from copilot.ingestion.chunker import build_segments, chunk_document
from copilot.ingestion.cleaner import clean_text
from copilot.ingestion.ids import make_content_hash, make_document_id, sha256_bytes
from copilot.ingestion.loaders import load_document
from copilot.ingestion.metadata import extract_metadata
from copilot.ingestion.models import IngestionResult, PageText
from copilot.ingestion.validators import IngestionError, safe_file_name, validate_upload
from copilot.logging_config import get_logger, redact
from copilot.utils import utc_now_iso

log = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates ingestion. Dependencies are injectable for testing."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Any | None = None,
        store: Any | None = None,
        repo: DocumentRepository | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if embedder is None:
            from copilot.embeddings.service import get_embedding_service

            embedder = get_embedding_service()
        if store is None:
            from copilot.vectorstore.pinecone_store import PineconeStore

            store = PineconeStore(self._settings)
        self._embedder = embedder
        self._store = store
        self._repo = repo or DocumentRepository(self._settings.sqlite_path)

    # ---- public API -----------------------------------------------------
    def ingest_bytes(
        self,
        filename: str,
        data: bytes,
        *,
        uploaded_by: str = "system",
        category: str | None = None,
        department: str | None = None,
        access_level: str | None = None,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest one file. Never raises for expected problems; returns a result instead."""
        try:
            name = safe_file_name(filename)
        except IngestionError as exc:
            return IngestionResult(status="failed", document_name=str(filename), message=str(exc), error=str(exc))
        document_id = make_document_id(name)
        existing = self._repo.get(document_id)

        # ---- prepare (non-destructive) ----
        try:
            file_type = validate_upload(name, data, self._settings)
            loaded = load_document(name, file_type, data)
            loaded = loaded.model_copy(
                update={"pages": [PageText(page_number=p.page_number, text=clean_text(p.text)) for p in loaded.pages]}
            )
            metadata, loaded = extract_metadata(
                loaded, category=category, department=department, access_level=access_level
            )
            segments = build_segments(loaded)
            if not segments:
                raise IngestionError("Document contains no extractable text (is it a scanned image?).")
            full_text = "\n\n".join(seg.text for seg in segments)
            content_hash = make_content_hash(
                full_text, category=metadata.category, department=metadata.department, access_level=metadata.access_level,
                status=metadata.status, superseded_by=metadata.superseded_by,
            )
        except IngestionError as exc:
            return self._record_prepare_failure(name, document_id, data, existing, uploaded_by, str(exc))
        except Exception as exc:  # noqa: BLE001 - unexpected parser failure
            log.exception("Unexpected error while preparing document", extra={"document_name": name})
            return self._record_prepare_failure(
                name, document_id, data, existing, uploaded_by, f"Unexpected error: {type(exc).__name__}"
            )

        # ---- duplicate detection ----
        if not force:
            if existing and existing.content_hash == content_hash and existing.status == "indexed":
                return IngestionResult(
                    status="skipped", document_name=name, document_id=document_id, version=existing.version,
                    chunk_count=existing.chunk_count, message="Unchanged: identical content is already indexed.",
                )
            duplicate = self._repo.get_by_hash(content_hash)
            if duplicate and duplicate.document_id != document_id:
                return IngestionResult(
                    status="skipped", document_name=name, document_id=document_id,
                    message=f"Duplicate content: already indexed as '{duplicate.document_name}'.",
                )

        version = existing.version + 1 if existing and existing.content_hash != content_hash else (
            existing.version if existing else 1
        )
        uploaded_at = utc_now_iso()
        self._repo.upsert(
            DocumentRecord(
                document_id=document_id, document_name=name, content_hash=content_hash,
                category=metadata.category, department=metadata.department, access_level=metadata.access_level,
                version=version, status="processing", file_type=file_type, file_size_bytes=len(data),
                page_count=loaded.page_count, chunk_count=0, uploaded_by=uploaded_by,
            )
        )
        record_id = self._repo.add_ingestion_record(document_id, version, content_hash)

        # ---- index ----
        try:
            self._save_source(document_id, name, data)
            chunks = chunk_document(
                document_id=document_id, document_name=name, version=version, uploaded_at=uploaded_at,
                segments=segments, metadata=metadata, settings=self._settings,
            )
            vectors = self._embedder.embed_documents([c.embedding_text for c in chunks])
            self._store.upsert(chunks, vectors)
            removed = self._store.delete_document(document_id, keep_version=version)
        except Exception as exc:  # noqa: BLE001 - external service/IO failure
            error = redact(f"{type(exc).__name__}: {exc}")[:300]
            log.error("Indexing failed", extra={"document_name": name, "error": error})
            self._repo.set_status(document_id, "failed", error=error)
            self._repo.finish_ingestion_record(record_id, "failed", error=error)
            return IngestionResult(
                status="failed", document_name=name, document_id=document_id, version=version,
                message="Indexing failed.", error=error,
            )

        self._repo.set_status(document_id, "indexed", chunk_count=len(chunks))
        self._repo.finish_ingestion_record(record_id, "indexed", chunk_count=len(chunks))
        log.info(
            "Document indexed",
            extra={"document_name": name, "version": version, "chunks": len(chunks), "stale_removed": removed},
        )
        return IngestionResult(
            status="indexed", document_name=name, document_id=document_id, version=version,
            chunk_count=len(chunks), message=f"Indexed {len(chunks)} chunks (version {version}).",
        )

    def ingest_path(self, path: Path | str, **kwargs: Any) -> IngestionResult:
        """Ingest a file from disk."""
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as exc:
            return IngestionResult(status="failed", document_name=p.name, message=str(exc), error=str(exc))
        return self.ingest_bytes(p.name, data, **kwargs)

    def ingest_directory(self, directory: Path | str | None = None, **kwargs: Any) -> list[IngestionResult]:
        """Ingest every supported file in a directory (non-recursive)."""
        folder = Path(directory) if directory else self._settings.kb_documents_dir
        files = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in self._settings.allowed_extensions)
        return [self.ingest_path(f, **kwargs) for f in files]

    def reindex(self, document_id: str) -> IngestionResult:
        """Re-embed and re-index a document from its stored source file."""
        record = self._repo.get(document_id)
        if record is None:
            return IngestionResult(status="failed", document_name=document_id, message="Document not found.", error="not_found")
        source = self._source_path(document_id, record.document_name)
        if not source.exists():
            msg = "Original file is not stored; re-upload the document instead."
            return IngestionResult(status="failed", document_name=record.document_name, document_id=document_id, message=msg, error=msg)
        return self.ingest_bytes(
            record.document_name, source.read_bytes(), uploaded_by=record.uploaded_by or "system",
            category=record.category, department=record.department, access_level=record.access_level, force=True,
        )

    def delete_document(self, document_id: str) -> bool:
        """Delete vectors, SQLite rows and the stored source file. True if the document existed."""
        record = self._repo.get(document_id)
        if record is None:
            return False
        self._store.delete_document(document_id)
        self._repo.delete(document_id)
        source = self._source_path(document_id, record.document_name)
        if source.exists():
            source.unlink()
        log.info("Document deleted", extra={"document_name": record.document_name})
        return True

    # ---- helpers ---------------------------------------------------------
    def _source_path(self, document_id: str, name: str) -> Path:
        return self._settings.uploads_dir / f"{document_id}{Path(name).suffix.lower()}"

    def _save_source(self, document_id: str, name: str, data: bytes) -> None:
        path = self._source_path(document_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _record_prepare_failure(
        self, name: str, document_id: str, data: bytes, existing: DocumentRecord | None, uploaded_by: str, error: str
    ) -> IngestionResult:
        """Record a pre-indexing failure without disturbing an already-indexed document."""
        raw_hash = sha256_bytes(data)
        if existing is None:
            self._repo.upsert(
                DocumentRecord(
                    document_id=document_id, document_name=name, content_hash=raw_hash, status="failed",
                    file_size_bytes=len(data), uploaded_by=uploaded_by, error=error,
                )
            )
        elif existing.status != "indexed":
            self._repo.set_status(document_id, "failed", error=error)
        record_id = self._repo.add_ingestion_record(document_id, existing.version if existing else 1, raw_hash)
        self._repo.finish_ingestion_record(record_id, "failed", error=error)
        log.warning("Ingestion rejected", extra={"document_name": name, "error": error})
        return IngestionResult(status="failed", document_name=name, document_id=document_id, message=error, error=error)
