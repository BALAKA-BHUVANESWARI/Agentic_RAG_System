"""Pydantic data models shared across the ingestion pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from copilot.config import AccessLevel

FileType = Literal["pdf", "docx", "txt"]


class PageText(BaseModel):
    """Raw text of one PDF page, or of a whole DOCX/TXT file (page_number=None)."""

    page_number: int | None = None
    text: str


class LoadedDocument(BaseModel):
    """Result of loading a file, before cleaning/chunking."""

    file_name: str
    file_type: FileType
    pages: list[PageText]

    @property
    def page_count(self) -> int | None:
        """Number of pages for PDFs; None for formats without pages."""
        return len(self.pages) if self.file_type == "pdf" else None


class DocumentMetadata(BaseModel):
    """Document-level metadata resolved from overrides, header lines and defaults."""

    title: str
    category: str = "general"
    department: str = "IT"
    access_level: AccessLevel = "internal"
    last_reviewed: str | None = None
    source_version: str | None = None
    status: Literal["current", "superseded"] = "current"
    superseded_by: str | None = None  # name of the newer document, when the header names one


class Segment(BaseModel):
    """A contiguous piece of text with its location (page and/or section)."""

    page_number: int | None = None
    section: str | None = None
    text: str


class Chunk(BaseModel):
    """One indexable chunk with all metadata required by the project spec."""

    chunk_id: str
    document_id: str
    document_name: str
    title: str
    chunk_index: int
    text: str
    category: str
    department: str
    access_level: AccessLevel
    page_number: int | None = None
    section: str | None = None
    uploaded_at: str
    version: int
    last_reviewed: str | None = None
    status: Literal["current", "superseded"] = "current"
    superseded_by: str | None = None

    @property
    def embedding_text(self) -> str:
        """Text that is embedded: title and section give the chunk context."""
        prefix = self.title if not self.section else f"{self.title} | {self.section}"
        return f"{prefix}\n{self.text}"


class IngestionResult(BaseModel):
    """Outcome of one ingestion attempt."""

    status: Literal["indexed", "skipped", "failed"]
    document_name: str
    document_id: str | None = None
    version: int | None = None
    chunk_count: int = 0
    message: str = ""
    error: str | None = None
