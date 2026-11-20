"""SQLite access for document metadata and ingestion history."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from copilot.db.connection import connect
from copilot.utils import utc_now_iso

DocStatus = Literal["pending", "processing", "indexed", "failed"]  # indexing status
Lifecycle = Literal["current", "superseded"]  # version status of the document itself


class DocumentRecord(BaseModel):
    """One row of the ``documents`` table."""

    document_id: str
    document_name: str
    content_hash: str
    category: str = "general"
    department: str = "IT"
    access_level: str = "internal"
    version: int = 1
    status: DocStatus = "pending"
    file_type: str | None = None
    file_size_bytes: int | None = None
    page_count: int | None = None
    chunk_count: int = 0
    uploaded_by: str | None = None
    uploaded_at: str = ""
    updated_at: str = ""
    error: str | None = None
    lifecycle: Lifecycle = "current"
    superseded_by: str | None = None


class IngestionRecord(BaseModel):
    """One row of the ``ingestion_records`` table."""

    id: int
    document_id: str
    version: int
    content_hash: str
    status: DocStatus
    chunk_count: int
    error: str | None
    started_at: str
    finished_at: str | None


_DOC_COLUMNS = list(DocumentRecord.model_fields)


class DocumentRepository:
    """Repository for documents and their ingestion history."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    # ---- documents ------------------------------------------------------
    def get(self, document_id: str) -> DocumentRecord | None:
        """Fetch a document by ID."""
        with connect(self._db_path) as conn:
            row = conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        """Fetch an indexed document with identical content hash (duplicate detection)."""
        with connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? AND status = 'indexed' LIMIT 1",
                (content_hash,),
            ).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def upsert(self, record: DocumentRecord) -> None:
        """Insert or fully update a document row (timestamps are set here)."""
        now = utc_now_iso()
        data = record.model_copy(update={"uploaded_at": now, "updated_at": now}).model_dump()
        cols = ", ".join(_DOC_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in _DOC_COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in _DOC_COLUMNS if c != "document_id")
        with connect(self._db_path) as conn:
            conn.execute(
                f"INSERT INTO documents ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(document_id) DO UPDATE SET {updates}",
                data,
            )

    def set_status(
        self,
        document_id: str,
        status: DocStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Update status (and optionally chunk count / error) of a document."""
        with connect(self._db_path) as conn:
            if chunk_count is None:
                conn.execute(
                    "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE document_id = ?",
                    (status, error, utc_now_iso(), document_id),
                )
            else:
                conn.execute(
                    "UPDATE documents SET status = ?, error = ?, chunk_count = ?, updated_at = ? "
                    "WHERE document_id = ?",
                    (status, error, chunk_count, utc_now_iso(), document_id),
                )

    def set_lifecycle(self, document_id: str, lifecycle: Lifecycle, superseded_by: str | None) -> None:
        """Record the version status of a document (no re-indexing)."""
        with connect(self._db_path) as conn:
            conn.execute(
                "UPDATE documents SET lifecycle = ?, superseded_by = ?, updated_at = ? WHERE document_id = ?",
                (lifecycle, superseded_by, utc_now_iso(), document_id),
            )

    def list(
        self, access_levels: list[str] | None = None, status: DocStatus | None = None
    ) -> list[DocumentRecord]:
        """List documents, optionally limited to access levels and/or a status."""
        query, params = "SELECT * FROM documents", []
        clauses: list[str] = []
        if access_levels is not None:
            if not access_levels:
                return []
            clauses.append(f"access_level IN ({', '.join('?' for _ in access_levels)})")
            params.extend(access_levels)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY document_name"
        with connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]

    def delete(self, document_id: str) -> bool:
        """Delete a document row (ingestion records cascade). Returns True if it existed."""
        with connect(self._db_path) as conn:
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cur.rowcount > 0

    # ---- ingestion history ---------------------------------------------
    def add_ingestion_record(self, document_id: str, version: int, content_hash: str) -> int:
        """Start an ingestion record in 'processing' state; returns its ID."""
        with connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO ingestion_records(document_id, version, content_hash, status, started_at) "
                "VALUES (?, ?, ?, 'processing', ?)",
                (document_id, version, content_hash, utc_now_iso()),
            )
            return int(cur.lastrowid)

    def finish_ingestion_record(
        self, record_id: int, status: DocStatus, *, chunk_count: int = 0, error: str | None = None
    ) -> None:
        """Close an ingestion record with its final status."""
        with connect(self._db_path) as conn:
            conn.execute(
                "UPDATE ingestion_records SET status = ?, chunk_count = ?, error = ?, finished_at = ? "
                "WHERE id = ?",
                (status, chunk_count, error, utc_now_iso(), record_id),
            )

    def list_ingestion_records(self, document_id: str) -> list[IngestionRecord]:
        """Ingestion history for one document, oldest first."""
        with connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_records WHERE document_id = ? ORDER BY id", (document_id,)
            ).fetchall()
        return [IngestionRecord(**dict(r)) for r in rows]
