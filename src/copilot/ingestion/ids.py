"""Stable identifiers and content hashing for documents and chunks."""
from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path


def normalize_name(name: str) -> str:
    """Normalize a file name for ID generation: drop directories, NFKC, trim, lowercase."""
    base = Path(name.replace("\\", "/")).name
    return unicodedata.normalize("NFKC", base).strip().lower()


def make_document_id(name: str) -> str:
    """Stable document ID: SHA-256 hex digest of the normalized file name."""
    return hashlib.sha256(normalize_name(name).encode("utf-8")).hexdigest()


def make_chunk_id(document_id: str, version: int, index: int) -> str:
    """Chunk ID format required by the project: ``<document_id>:v<version>:<index>``."""
    return f"{document_id}:v{version}:{index}"


def make_content_hash(
    text: str, *, category: str, department: str, access_level: str, status: str = "current", superseded_by: str | None = None
) -> str:
    """Hash of cleaned text plus the metadata that changes how a document is retrieved.

    Including metadata means changing only the access level, category or superseded status
    triggers a re-index, while re-uploading identical content is detected as a duplicate.
    The status is added only for superseded documents, so hashes of current documents (all of
    them before the status header existed) do not change.
    """
    parts = [text, category, department, access_level]
    if status != "current":
        parts += [status, superseded_by or ""]
    payload = "\x1f".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()
