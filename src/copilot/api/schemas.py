"""Pydantic request/response models for the HTTP API (the public contract)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from copilot.audit.logger import AuditRecord

AccessLevelName = Literal["public", "internal", "restricted"]
LifecycleName = Literal["current", "superseded"]


class ErrorResponse(BaseModel):
    """Uniform error body."""

    detail: str
    request_id: str | None = None


# ---- chat -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """An employee question."""

    question: str = Field(min_length=1)
    session_id: str | None = Field(default=None, max_length=64)
    external_search_allowed: bool = Field(default=True, description="Set false to forbid public web search for this request.")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class CitationOut(BaseModel):
    """A citation resolved against real evidence."""

    ref: int
    kind: Literal["private", "external"]
    label: str
    location: str
    url: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page_number: int | None = None
    section: str | None = None


class TraceEventOut(BaseModel):
    """One real graph execution event."""

    node: str
    started_at: str
    duration_ms: float
    summary: str = ""
    decision: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Answer, citations and the decision trace."""

    request_id: str
    answer: str = Field(description="Text to show the user (warning + body + Sources).")
    abstained: bool
    abstention_reason: str | None = None
    used_external: bool = False
    citations: list[CitationOut] = Field(default_factory=list)
    path: list[str] = Field(description="Graph nodes in the order they actually ran.")
    trace: list[TraceEventOut] = Field(default_factory=list)
    classification: dict[str, Any] | None = None
    private_grade: dict[str, Any] | None = None
    retrieval_attempts: list[dict[str, Any]] = Field(default_factory=list)
    rewritten_queries: list[str] = Field(default_factory=list)
    retry_count: int = 0
    external_search_triggered: bool = False
    external_query: str | None = None
    external_results: list[dict[str, Any]] = Field(default_factory=list)
    external_grade: dict[str, Any] | None = None
    external_blocked_reason: str | None = None
    validation: dict[str, Any] | None = None
    tokens: int = 0
    llm_calls: int = 0
    tavily_calls: int = 0
    latency_ms: float


# ---- documents ------------------------------------------------------------------------
class DocumentOut(BaseModel):
    """Document metadata visible to the caller."""

    document_id: str
    document_name: str
    category: str
    department: str
    access_level: str
    version: int
    status: str
    file_type: str | None = None
    page_count: int | None = None
    chunk_count: int = 0
    lifecycle: LifecycleName = Field(default="current", description="Version status: 'superseded' = replaced by a newer document.")
    superseded_by: str | None = Field(default=None, description="Name of the replacing document, when known.")
    uploaded_at: str = ""
    updated_at: str = ""
    uploaded_by: str | None = Field(default=None, description="Admins only.")
    error: str | None = Field(default=None, description="Admins only.")


class DocumentListResponse(BaseModel):
    """Documents the caller is authorised to see."""

    role: str
    count: int
    documents: list[DocumentOut]


class IngestionOut(BaseModel):
    """Outcome of an upload or re-index."""

    status: Literal["indexed", "skipped", "failed"]
    document_name: str
    document_id: str | None = None
    version: int | None = None
    chunk_count: int = 0
    message: str = ""


class DeleteResponse(BaseModel):
    """Result of a document deletion."""

    document_id: str
    deleted: bool


# ---- feedback / audit -------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    """Thumbs up/down on an answer."""

    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal["positive", "negative"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Feedback receipt."""

    feedback_id: int
    request_id: str
    rating: str


class AuditResponse(AuditRecord):
    """Audit record plus any feedback given on it."""

    feedback: list[dict[str, Any]] = Field(default_factory=list)


# ---- health -----------------------------------------------------------------------------
class DependencyStatus(BaseModel):
    """Status of one dependency."""

    name: str
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    """Application and dependency health (no secrets)."""

    status: Literal["ok", "degraded"]
    dependencies: list[DependencyStatus]


# ---- evaluation -------------------------------------------------------------------------
class EvaluationRunRequest(BaseModel):
    """Parameters for an evaluation run (the run starts in the background)."""

    mode: Literal["agentic", "baseline", "both"] = "both"
    limit: int | None = Field(default=None, ge=1, description="Run only the first N cases (after the other filters).")
    per_category: int | None = Field(default=None, ge=1, description="Run only the first N cases of each category.")
    categories: list[str] | None = None
    use_judge: bool = Field(default=True, description="Also run the LLM judge (faithfulness / relevance). Costs extra tokens.")
    resume_run_id: str | None = Field(default=None, max_length=64, description="Resume this earlier run (its saved options are reused).")

    @field_validator("categories")
    @classmethod
    def _known_categories(cls, v: list[str] | None) -> list[str] | None:
        from copilot.evaluation.schemas import CATEGORIES

        bad = sorted(set(v or []) - set(CATEGORIES))
        if bad:
            raise ValueError(f"unknown categories {bad}; valid: {CATEGORIES}")
        return v


class EvaluationRunInfo(BaseModel):
    """Status of one evaluation run."""

    run_id: str
    status: str  # running | cancelling | completed | cancelled | stopped | failed | interrupted
    mode: str | None = None
    dataset_path: str | None = None
    num_cases: int | None = None
    use_judge: bool | None = None
    done: int = 0
    total: int = 0
    message: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None
