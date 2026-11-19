"""Models for retrieved chunks and retrieval results."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store, with its relevance score."""

    chunk_id: str
    document_id: str
    document_name: str
    title: str = ""
    category: str = ""
    department: str = ""
    access_level: str = "internal"
    page_number: int | None = None
    section: str | None = None
    version: int = 1
    uploaded_at: str = ""
    last_reviewed: str | None = None
    text: str = ""
    score: float = 0.0  # vector similarity (cosine) from Pinecone
    rerank_score: float | None = None  # cross-encoder score, set only when reranking ran

    @classmethod
    def from_match(cls, match_id: str, score: float, metadata: dict[str, Any]) -> "RetrievedChunk":
        """Build from a Pinecone match (id, score, metadata)."""
        md = dict(metadata or {})
        page = md.get("page_number")
        return cls(
            chunk_id=md.get("chunk_id", match_id),
            document_id=md.get("document_id", ""),
            document_name=md.get("document_name", ""),
            title=md.get("title", ""),
            category=md.get("category", ""),
            department=md.get("department", ""),
            access_level=md.get("access_level", "restricted"),  # unknown level -> most restrictive
            page_number=int(page) if page is not None else None,
            section=md.get("section"),
            version=int(md.get("version", 1)),
            uploaded_at=md.get("uploaded_at", ""),
            last_reviewed=md.get("last_reviewed"),
            text=md.get("text", ""),
            score=float(score),
        )

    @property
    def location(self) -> str:
        """Human-readable location used in citations."""
        parts = []
        if self.page_number is not None:
            parts.append(f"page {self.page_number}")
        if self.section:
            parts.append(f"section '{self.section}'")
        return ", ".join(parts) or "document"


class RetrievalFilters(BaseModel):
    """Optional metadata filters (access filtering is always applied on top)."""

    category: str | None = None
    department: str | None = None
    document_ids: list[str] | None = None


class RetrievalResult(BaseModel):
    """Outcome of a retrieval call, including why it may be empty."""

    query: str
    role: str
    top_k: int
    status: Literal["ok", "empty", "no_access"]
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    dropped_unauthorized: int = 0
    dropped_low_score: int = 0
    reranked: bool = False
