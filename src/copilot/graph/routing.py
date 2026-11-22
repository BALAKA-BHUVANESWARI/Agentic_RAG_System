"""Conditional-edge routing functions (pure; unit-testable without a graph).

Invariants enforced here:
* Private retrieval always runs first for in-scope questions. An ``out_of_scope`` verdict is not trusted on its own
  (the model once called a company-password-policy question out of scope, finding 9, X18): retrieval still runs once as a
  check. Nothing above ``RETRIEVAL_MIN_SCORE`` means the decline stands; otherwise the grader decides, with no rewrite
  retries and no web search, because only ``external_current`` / ``both`` can open external search.
* External search needs: insufficient private evidence (or a genuine supplement need),
  AND search allowed (config + request), AND no sensitive content, AND the CLASSIFIER says the
  question needs current public information (intent ``external_current`` or ``both``).
  The grader's ``recommended_action`` is a signal for the trace only: it can never open the
  door to web search by itself (an undocumented company question must abstain).
* Every loop is bounded by MAX_REWRITE_RETRIES / MAX_VALIDATION_REVISIONS.
* Retrieved content never reaches this module: it only reads structured fields.
"""
from __future__ import annotations

from typing import Literal

from copilot.config import Settings
from copilot.graph.state import CopilotState

ANSWER = "answer_generator"
RETRIEVER = "private_retriever"
GRADER = "private_evidence_grader"
REWRITER = "query_rewriter"
EXTERNAL = "external_search"
AUDIT = "audit_logger"


def _blocked(state: CopilotState) -> bool:
    return bool(state.get("fatal_error") or state.get("force_abstain"))


def external_permitted(state: CopilotState, settings: Settings) -> bool:
    """Policy gate: may a public search happen for this request at all?"""
    classification = state.get("classification") or {}
    return bool(
        settings.external_search_enabled
        and state.get("external_search_allowed", True)
        and not classification.get("contains_sensitive")
    )


def external_eligible(state: CopilotState, settings: Settings) -> bool:
    """After insufficient private evidence: permitted AND the classifier found a public-information angle.

    A failed retrieval alone is NOT enough: company-specific questions that simply are not
    documented are abstained, not sent to the web. The grader's ``recommended_action`` is
    deliberately ignored here: an LLM grader once recommended a web search for an internal
    hardware-request question that the classifier had (correctly) marked ``internal_it``.
    """
    if not external_permitted(state, settings):
        return False
    intent = (state.get("classification") or {}).get("intent")
    return intent in {"external_current", "both"}


def wants_external_supplement(state: CopilotState, settings: Settings) -> bool:
    """Scenario C: private evidence suffices but BOTH the classifier and the grader say public info is also needed."""
    intent = (state.get("classification") or {}).get("intent")
    grade = state.get("private_grade") or {}
    return bool(intent == "both" and grade.get("needs_external_supplement") and external_permitted(state, settings))


def _out_of_scope(state: CopilotState) -> bool:
    return (state.get("classification") or {}).get("intent") == "out_of_scope"


def route_after_classifier(state: CopilotState) -> Literal["private_retriever", "answer_generator"]:
    """Every question, even one called out of scope, gets a private retrieval check first (never web search)."""
    return ANSWER if _blocked(state) else RETRIEVER


def route_after_retriever(state: CopilotState) -> Literal["private_evidence_grader", "answer_generator"]:
    """A retrieval outage abstains instead of grading nothing; an out-of-scope verdict with nothing retrieved is declined."""
    if _blocked(state) or (_out_of_scope(state) and not state.get("retrieved_chunks")):
        return ANSWER
    return GRADER


def route_after_private_grader(
    state: CopilotState, settings: Settings
) -> Literal["answer_generator", "query_rewriter", "external_search"]:
    """Decide between answering, rewriting (bounded), external search, or abstaining."""
    if _blocked(state) or _out_of_scope(state):
        return ANSWER  # out of scope: the grader only decides whether the documents do cover it; no rewrite, no search
    grade = state.get("private_grade") or {}
    if grade.get("effective_sufficient"):
        return EXTERNAL if wants_external_supplement(state, settings) else ANSWER
    if state.get("retry_count", 0) < settings.max_rewrite_retries and not state.get("rewrite_failed"):
        return REWRITER
    return EXTERNAL if external_eligible(state, settings) else ANSWER


def route_after_rewriter(
    state: CopilotState, settings: Settings
) -> Literal["private_retriever", "external_search", "answer_generator"]:
    """Retry retrieval with the rewritten query, or move on if rewriting failed."""
    if _blocked(state):
        return ANSWER
    if state.get("rewrite_failed"):
        return EXTERNAL if external_eligible(state, settings) else ANSWER
    return RETRIEVER


def route_after_validator(state: CopilotState) -> Literal["answer_generator", "audit_logger"]:
    """Revise once if the validator asked for it (the node enforces the revision limit)."""
    return ANSWER if state.get("needs_revision") else AUDIT
