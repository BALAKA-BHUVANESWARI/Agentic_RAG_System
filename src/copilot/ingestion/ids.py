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


def make_content_hash(text: str, *, category: str, department: str, access_level: str) -> str:
    """Hash of cleaned text plus access-relevant metadata.

    Including metadata means changing only the access level or category triggers a
    re-index, while re-uploading identical content is detected as a duplicate.
    """
    payload = "\x1f".join([text, category, department, access_level])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()
