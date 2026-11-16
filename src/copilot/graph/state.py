"""Typed LangGraph state and the trace event model.

The state is flat and JSON-serialisable (plain dicts/lists, no custom objects).
Nodes return PARTIAL updates; list/dict fields with reducers are appended/merged.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """One real execution event, appended by each node. The UI renders these as-is."""

    node: str
    started_at: str
    duration_ms: float
    summary: str = ""
    decision: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def add_usage(left: dict | None, right: dict | None) -> dict:
    """Reducer: sum LLM usage dicts (prompt/completion/total tokens and calls)."""
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    for part in (left or {}, right or {}):
        for key in out:
            out[key] += int(part.get(key, 0) or 0)
    return out


class CopilotState(TypedDict, total=False):
    """Graph state. Keys without a reducer are overwritten by the latest node that sets them."""

    # --- request ---
    request_id: str
    session_id: str
    user_id: str
    role: str
    question: str
    external_search_allowed: bool
    started_perf: float

    # --- classification ---
    classification: dict

    # --- private retrieval / grading / rewriting ---
    current_query: str
    retry_count: int
    rewrite_failed: bool
    rewritten_queries: Annotated[list[str], operator.add]
    retrieved_chunks: list[dict]
    retrieval_attempts: Annotated[list[dict], operator.add]
    private_grade: dict

    # --- external search ---
    external_triggered: bool
    external_query: str
    external_blocked_reason: str
    external_results: list[dict]
    external_grade: dict
    accepted_external: list[dict]
    tavily_calls: Annotated[int, operator.add]

    # --- generation / validation ---
    evidence: list[dict]
    answer: dict
    revision_count: int
    validator_feedback: str
    needs_revision: bool
    validation: dict

    # --- control ---
    force_abstain: dict
    fatal_error: str
    llm_usage: Annotated[dict, add_usage]
    trace: Annotated[list[dict], operator.add]
