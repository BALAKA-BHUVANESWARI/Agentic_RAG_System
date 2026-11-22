"""Unit tests for the audit logger, routing functions and graph schemas."""
from __future__ import annotations

import json

import pytest

from copilot.audit.logger import AuditLogger
from copilot.graph import routing
from copilot.graph.schemas import ExternalResultGrade, PrivateGrade, QueryClassification
from copilot.graph.state import add_usage


# ---- schemas ---------------------------------------------------------------------------------
def test_confidence_is_clamped_and_percentages_tolerated():
    assert PrivateGrade(sufficient=True, confidence=85, recommended_action="answer").confidence == 0.85
    assert PrivateGrade(sufficient=True, confidence=1.7, recommended_action="answer").confidence == 0.017
    assert PrivateGrade(sufficient=True, confidence=-3, recommended_action="answer").confidence == 0.0
    assert QueryClassification(intent="internal_it", confidence=0.4).confidence == 0.4
    assert ExternalResultGrade(ref=1, relevance=90, credibility=0.5).relevance == 0.9


def test_invalid_intent_is_rejected():
    with pytest.raises(ValueError):
        QueryClassification(intent="weather")


def test_usage_reducer_sums_and_tolerates_none():
    assert add_usage(None, {"total_tokens": 5, "calls": 1}) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 5, "calls": 1}
    assert add_usage({"total_tokens": 2, "calls": 1}, {"total_tokens": 3, "calls": 2})["calls"] == 3


# ---- audit ------------------------------------------------------------------------------------
def _state(**kw):
    base = dict(
        request_id="r1", user_id="u", role="employee", question="How do I use VPN?",
        classification={"intent": "internal_it"},
        retrieval_attempts=[{"attempt": 0, "query": "q", "chunk_ids": ["c1", "c2"], "document_ids": ["d1"], "scores": [{"chunk_id": "c1", "score": 0.5}]}],
        rewritten_queries=["q2"], external_triggered=True, external_results=[{"url": "https://learn.microsoft.com/a"}],
        private_grade={"sufficient": False}, answer={"rendered": "Answer [1]", "citations": [{"ref": 1}], "abstained": False},
        validation={"passed": True}, trace=[{"node": "query_classifier"}],
    )
    base.update(kw)
    return base


def test_audit_roundtrip(tmp_path):
    audit = AuditLogger(tmp_path / "a.db")
    audit.record_state(_state(), execution_time_ms=123.4)
    rec = audit.get("r1")
    assert rec.intent == "internal_it" and rec.retrieved_chunk_ids == ["c1", "c2"] and rec.retrieved_document_ids == ["d1"]
    assert rec.rewritten_query == "q2" and rec.external_search_triggered is True and rec.external_urls == ["https://learn.microsoft.com/a"]
    assert rec.final_answer == "Answer [1]" and rec.citations == [{"ref": 1}] and rec.execution_time_ms == 123.4
    assert rec.grading_results["private"] == {"sufficient": False} and rec.trace == [{"node": "query_classifier"}]
    assert rec.retrieval_scores[0]["chunk_id"] == "c1" and rec.abstained is False and audit.get("nope") is None


def test_audit_redacts_secrets_in_question_and_answer(tmp_path):
    audit = AuditLogger(tmp_path / "a.db")
    audit.record_state(
        _state(question="my password is hunter2 and key gsk_abcdefghij1234567", answer={"rendered": "token: abc999", "abstained": True, "abstention_reason": "no_evidence"}),
        execution_time_ms=1,
    )
    rec = audit.get("r1")
    for secret in ("hunter2", "gsk_abcdefghij", "abc999"):
        assert secret not in rec.question and secret not in (rec.final_answer or "")
    assert rec.abstained and rec.abstention_reason == "no_evidence"


_CHUNK_ID = "e9dedc92d7347ec41418059ecb6e7c215401edbb55a23d5bb23cdfe4016d33f8:v1:2"
_LONG_URL = "https://www.businessinsider.com/microsoft-teams-has-worldwide-outage-2021-4?IR=T&r=US"


def test_audit_redacts_a_typed_secret_inside_the_stored_trace(tmp_path):
    """DEMO.md S6 found the raw password in trace[].data.query; the whole trace is now redacted."""
    trace = [
        {"node": "private_retriever", "summary": "attempt 0", "data": {"query": "my password is hunter2, why does Teams fail?", "documents": ["Laptop.txt (0.41)"]}},
        {"node": "query_rewriter", "data": {"rewritten_query": "api_key=abc12345 Teams failure", "added_terms": ["teams"]}},
    ]
    audit = AuditLogger(tmp_path / "a.db")
    audit.record_state(_state(trace=trace, question="my password is hunter2, why does Teams fail?"), execution_time_ms=1)
    stored = audit.get("r1")
    blob = json.dumps(stored.trace)
    assert "hunter2" not in blob and "abc12345" not in blob
    assert stored.trace[0]["node"] == "private_retriever" and stored.trace[0]["data"]["documents"] == ["Laptop.txt (0.41)"]


def test_audit_masks_a_long_typed_token_in_the_trace_but_keeps_ids_and_urls(tmp_path):
    long_token = "A1b2C3d4" * 6  # 48 characters with no keyword in front of it
    trace = [
        {"node": "private_retriever", "data": {"query": f"why does {long_token} fail", "supporting_chunks": [_CHUNK_ID]}},
        {"node": "external_search", "data": {"urls": [_LONG_URL]}},
    ]
    audit = AuditLogger(tmp_path / "a.db")
    audit.record_state(_state(trace=trace), execution_time_ms=1)
    blob = json.dumps(audit.get("r1").trace)
    assert long_token not in blob
    assert _CHUNK_ID in blob and _LONG_URL in blob


def test_audit_redacts_secrets_in_grading_and_validation_results(tmp_path):
    audit = AuditLogger(tmp_path / "a.db")
    audit.record_state(
        _state(private_grade={"reasoning": "user wrote password: hunter2"}, validation={"reasoning": "token: abc999 is not evidence"}),
        execution_time_ms=1,
    )
    stored = audit.get("r1")
    blob = json.dumps([stored.grading_results, stored.validation_result])
    assert "hunter2" not in blob and "abc999" not in blob


def test_audit_keeps_long_urls_in_the_stored_answer_but_still_masks_credentials(tmp_path):
    """The old full redaction turned https://learn.microsoft.com/... into https://learn.microsoft.[REDACTED] in stored answers."""
    answer = {"rendered": f"External information: see [1].\n\nSources:\n[1] Title - {_LONG_URL}\n\nnote token: abc999", "abstained": False}
    audit = AuditLogger(tmp_path / "a.db")
    audit.record_state(_state(answer=answer), execution_time_ms=1)
    stored = audit.get("r1").final_answer
    assert _LONG_URL in stored and "abc999" not in stored


def test_audit_never_raises(tmp_path):
    AuditLogger(tmp_path / "a.db").record_state({"question": "no request id"}, execution_time_ms=1)


# ---- routing -----------------------------------------------------------------------------------
def _st(intent="internal_it", sensitive=False, **kw):
    return {"classification": {"intent": intent, "contains_sensitive": sensitive}, "retry_count": 0, **kw}


def _grade(effective=False, action="rewrite_query", supplement=False):
    return {"effective_sufficient": effective, "recommended_action": action, "needs_external_supplement": supplement}


def test_classifier_routing():
    assert routing.route_after_classifier(_st("internal_it")) == "private_retriever"
    assert routing.route_after_classifier(_st("external_current")) == "private_retriever"  # private-first, always
    assert routing.route_after_classifier(_st("out_of_scope")) == "private_retriever"  # the verdict is checked, not trusted
    assert routing.route_after_classifier({**_st(), "fatal_error": "x"}) == "answer_generator"


def test_out_of_scope_verdict_is_declined_only_when_nothing_relevant_is_retrieved():
    chunk = {"chunk_id": "c1"}
    assert routing.route_after_retriever(_st("out_of_scope", retrieved_chunks=[])) == "answer_generator"
    assert routing.route_after_retriever(_st("out_of_scope")) == "answer_generator"
    assert routing.route_after_retriever(_st("out_of_scope", retrieved_chunks=[chunk])) == "private_evidence_grader"
    assert routing.route_after_retriever(_st("internal_it", retrieved_chunks=[])) == "private_evidence_grader"  # unchanged
    assert routing.route_after_retriever(_st("out_of_scope", retrieved_chunks=[chunk], fatal_error="x")) == "answer_generator"


def test_out_of_scope_never_rewrites_or_searches(make_settings):
    s = make_settings()
    for grade in (_grade(False, "external_search"), _grade(False, "rewrite_query"), _grade(True, "answer", supplement=True)):
        st = _st("out_of_scope", private_grade=grade)
        assert routing.route_after_private_grader(st, s) == "answer_generator"


def test_sufficient_private_evidence_never_triggers_search_even_for_external_intent(make_settings):
    s = make_settings()
    for intent in ("internal_it", "external_current", "both"):
        st = _st(intent, private_grade=_grade(True, "answer", supplement=False))
        assert routing.route_after_private_grader(st, s) == "answer_generator"


def test_supplement_needs_both_signals_and_permission(make_settings):
    s = make_settings()
    both = _st("both", private_grade=_grade(True, "answer", supplement=True))
    assert routing.route_after_private_grader(both, s) == "external_search"
    assert routing.route_after_private_grader(_st("internal_it", private_grade=_grade(True, "answer", True)), s) == "answer_generator"
    assert routing.route_after_private_grader(_st("both", True, private_grade=_grade(True, "answer", True)), s) == "answer_generator"
    assert routing.route_after_private_grader({**both, "external_search_allowed": False}, s) == "answer_generator"
    assert routing.route_after_private_grader(both, make_settings(external_search_enabled=False)) == "answer_generator"


def test_insufficient_evidence_retries_then_decides_on_external(make_settings):
    s = make_settings(max_rewrite_retries=2)
    for n, expected in ((0, "query_rewriter"), (1, "query_rewriter"), (2, "external_search")):
        st = _st("external_current", retry_count=n, private_grade=_grade())
        assert routing.route_after_private_grader(st, s) == expected


def test_failed_retrieval_alone_does_not_justify_external_search(make_settings):
    s = make_settings(max_rewrite_retries=1)
    undocumented = _st("internal_it", retry_count=1, private_grade=_grade(action="abstain"))
    assert routing.route_after_private_grader(undocumented, s) == "answer_generator"
    # The grader's recommendation alone must NOT open the door to web search (regression: eval case I02).
    grader_says_public = _st("internal_it", retry_count=1, private_grade=_grade(action="external_search"))
    assert routing.route_after_private_grader(grader_says_public, s) == "answer_generator"
    assert routing.route_after_rewriter(_st("internal_it", rewrite_failed=True, private_grade=_grade(action="external_search")), s) == "answer_generator"
    # ...while a classifier-flagged public question still reaches web search, whatever the grader recommends.
    for intent in ("external_current", "both"):
        for action in ("abstain", "rewrite_query", "external_search"):
            st = _st(intent, retry_count=1, private_grade=_grade(action=action))
            assert routing.route_after_private_grader(st, s) == "external_search"


@pytest.mark.parametrize("override", [{"classification": {"intent": "external_current", "contains_sensitive": True}},
                                      {"external_search_allowed": False}])
def test_external_blocked_by_sensitivity_or_request_flag(make_settings, override):
    st = {**_st("external_current", retry_count=2, private_grade=_grade()), **override}
    assert routing.route_after_private_grader(st, make_settings(max_rewrite_retries=2)) == "answer_generator"


def test_rewriter_routing(make_settings):
    s = make_settings()
    assert routing.route_after_rewriter(_st(private_grade=_grade()), s) == "private_retriever"
    assert routing.route_after_rewriter(_st("external_current", rewrite_failed=True, private_grade=_grade()), s) == "external_search"
    assert routing.route_after_rewriter(_st("internal_it", rewrite_failed=True, private_grade=_grade()), s) == "answer_generator"


def test_grader_failure_abstains_instead_of_searching(make_settings):
    st = _st("external_current", force_abstain={"reason": "grading_unavailable"}, private_grade=_grade())
    assert routing.route_after_private_grader(st, make_settings()) == "answer_generator"
    assert routing.route_after_retriever({"force_abstain": {"reason": "retrieval_unavailable"}}) == "answer_generator"
    assert routing.route_after_retriever({}) == "private_evidence_grader"


def test_validator_routing():
    assert routing.route_after_validator({"needs_revision": True}) == "answer_generator"
    assert routing.route_after_validator({"needs_revision": False}) == "audit_logger"
    assert routing.route_after_validator({}) == "audit_logger"
