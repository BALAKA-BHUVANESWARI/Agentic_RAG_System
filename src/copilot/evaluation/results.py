"""Per-case result records (JSON-serialisable; written to disk as the run progresses)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedItem(BaseModel):
    """One retrieved chunk, in rank order."""

    document_name: str
    chunk_id: str
    score: float | None = None
    text: str = ""


class RetrievalCaseResult(BaseModel):
    """Ranked retrieval for one question plus its retrieval metrics."""

    case_id: str
    category: str
    items: list[RetrievedItem] = Field(default_factory=list)
    scores: dict[str, float | None] = Field(default_factory=dict)


class JudgeScores(BaseModel):
    """LLM-as-judge output (0-1 scales)."""

    faithfulness: float | None = None
    relevance: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


class CaseResult(BaseModel):
    """Everything recorded about one system's answer to one question."""

    case_id: str
    category: str
    mode: Literal["baseline", "agentic"]
    question: str
    role: str = "employee"
    request_id: str | None = None
    answer: str = ""
    abstained: bool = False
    abstention_reason: str | None = None
    used_external: bool = False
    cited_documents: list[str] = Field(default_factory=list)
    external_urls: list[str] = Field(default_factory=list)
    path: list[str] = Field(default_factory=list)
    external_triggered: bool = False
    tavily_calls: int = 0
    retry_count: int = 0
    rewritten_queries: list[str] = Field(default_factory=list)
    grade_sufficient: bool | None = None
    grade_confidence: float | None = None
    validation_passed: bool | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    error: str | None = None
    judge: JudgeScores | None = None
    scores: dict[str, Any] = Field(default_factory=dict)
