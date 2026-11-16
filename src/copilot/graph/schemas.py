"""Structured LLM outputs used by graph nodes (validated with Pydantic)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["internal_it", "external_current", "both", "out_of_scope"]
RecommendedAction = Literal["answer", "rewrite_query", "external_search", "abstain"]


def _unit_interval(value: object) -> object:
    """Accept 0-1 floats and tolerate percentages (e.g. 85 -> 0.85); clamp the rest."""
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1.0 and v <= 100.0:
            v /= 100.0
        return min(1.0, max(0.0, v))
    return value


class QueryClassification(BaseModel):
    """Output of the query classifier node."""

    intent: Intent
    needs_current_public_info: bool = False
    contains_sensitive: bool = False
    sensitive_reason: str | None = None
    public_search_query: str | None = None
    confidence: float = 0.5
    reasoning: str = ""

    _clamp = field_validator("confidence", mode="before")(_unit_interval)


class PrivateGrade(BaseModel):
    """Output of the private evidence grader node.

    ``confidence`` is a decision signal, NOT a calibrated probability.
    """

    sufficient: bool
    confidence: float = 0.0
    supporting_refs: list[int] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    contradiction_detected: bool = False
    contradiction_details: str | None = None
    outdated_or_incomplete: bool = False
    needs_external_supplement: bool = False
    reasoning: str = ""
    recommended_action: RecommendedAction = "abstain"

    _clamp = field_validator("confidence", mode="before")(_unit_interval)


class QueryRewrite(BaseModel):
    """Output of the query rewriter node."""

    rewritten_query: str
    missing_concepts: list[str] = Field(default_factory=list)
    added_terms: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ExternalResultGrade(BaseModel):
    """Grade of one external search result (referenced by its evidence id)."""

    ref: int
    relevance: float = 0.0
    credibility: float = 0.0
    freshness: Literal["current", "stale", "unknown"] = "unknown"
    supports_answer: bool = False
    accept: bool = False
    reason: str = ""

    _clamp = field_validator("relevance", "credibility", mode="before")(_unit_interval)


class ExternalGrade(BaseModel):
    """Output of the external evidence grader node."""

    results: list[ExternalResultGrade] = Field(default_factory=list)
    sufficient: bool = False
    confidence: float = 0.0
    contradiction_detected: bool = False
    reasoning: str = ""

    _clamp = field_validator("confidence", mode="before")(_unit_interval)


class GroundingCheck(BaseModel):
    """Output of the final answer validator's grounding check."""

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    conflicts_communicated: bool = True
    external_clearly_separated: bool = True
    reasoning: str = ""
