"""Data models for evidence, citations and generated answers."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from copilot.llm.client import LLMUsage

AbstentionReason = Literal[
    "no_evidence",
    "insufficient_evidence",
    "uncited_answer",
    "invalid_llm_output",
    "llm_unavailable",
    "retrieval_unavailable",
    "out_of_scope",
    "grading_unavailable",
    "validation_failed",
    "internal_error",
]


class EvidenceItem(BaseModel):
    """One numbered piece of evidence shown to the LLM (private chunk or external page)."""

    ref: int  # 1-based number the model must cite
    kind: Literal["private", "external"]
    label: str  # document name or page title
    location: str  # "page 3, section 'X'" or the page domain
    text: str
    url: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    page_number: int | None = None
    section: str | None = None
    last_reviewed: str | None = None  # private docs: review date from the document header
    published_date: str | None = None  # external pages: publication date if known


class Citation(BaseModel):
    """A citation resolved against real evidence (built by code, not by the LLM)."""

    ref: int
    kind: Literal["private", "external"]
    label: str
    location: str
    url: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    page_number: int | None = None
    section: str | None = None


class GeneratedAnswer(BaseModel):
    """Result of answer generation (also used for abstentions)."""

    answer: str = ""  # body with [n] markers; empty when abstained
    rendered: str  # what the user sees: warning + body + Sources
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None
    missing_information: str | None = None
    used_external: bool = False
    invalid_markers_removed: list[int] = Field(default_factory=list)
    uncited_lines: list[str] = Field(default_factory=list)
    error: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model: str = ""
