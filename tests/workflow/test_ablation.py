"""Ablation switches (AGENT_USE_GRADER / AGENT_USE_VALIDATOR): what each one changes in the REAL graph, and what it must not.

These switches exist only so the evaluation can measure what each agent step buys. The defaults keep the full workflow,
and no switch may weaken a routing invariant (private-first, no web search without graded insufficiency).
"""
from __future__ import annotations

from fakes import FakeSearch

from copilot.vectorstore.models import RetrievalResult
from workflow.test_scenarios import CLS_EXTERNAL, CLS_INTERNAL, GRADE_OK, OK_CHECK, PATH_A, TEAMS_RESULT, answer_draft

BAD_CHECK = dict(grounded=False, unsupported_claims=["an invented step"])


class EmptyRetriever:
    """Knowledge base that returns nothing for every query."""

    def retrieve(self, query, role):
        return RetrievalResult(query=query, role="employee", top_k=5, status="empty")


def test_defaults_keep_the_full_workflow(make_settings):
    s = make_settings()
    assert s.agent_use_grader is True and s.agent_use_validator is True


def test_grader_off_makes_no_llm_grading_call_and_answers_from_the_retrieved_chunks(make_agent, kb):
    h = make_agent(settings_overrides={"agent_use_grader": False}, QueryClassification=[CLS_INTERNAL],
                   AnswerDraft=[answer_draft("Open the SecureLink client and connect [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer("How do I connect to the company VPN from home?")

    assert r.path == PATH_A and h.llm.count("PrivateGrade") == 0
    assert h.llm.names() == ["QueryClassification", "AnswerDraft", "GroundingCheck"]
    assert r.private_grade["skipped"] is True and r.private_grade["effective_sufficient"] is True
    assert r.private_grade["selected_chunk_ids"]                              # the retrieved chunks reached the generator
    assert not r.answer.abstained and r.answer.citations[0].kind == "private" and h.search.call_count == 0
    grader_event = next(e for e in r.trace if e.node == "private_evidence_grader")
    assert "skipped" in grader_event.summary and grader_event.data["ablation"] == "agent_use_grader=false"


def test_grader_off_still_abstains_when_nothing_was_retrieved(make_agent, kb):
    h = make_agent(retriever=EmptyRetriever(), settings_overrides={"agent_use_grader": False, "max_rewrite_retries": 0},
                   QueryClassification=[CLS_INTERNAL])
    r = h.agent.answer("How do I connect to the company VPN from home?")
    assert r.answer.abstained and h.llm.names() == ["QueryClassification"] and h.search.call_count == 0
    assert not r.private_grade.get("skipped")                                 # the empty-retrieval guard decided, not the switch


def test_grader_off_cannot_open_the_web_door(make_agent, kb):
    """A web question whose chunks were retrieved is answered privately: with no 'insufficient' verdict there is no fallback."""
    search = FakeSearch([TEAMS_RESULT])
    h = make_agent(search=search, settings_overrides={"agent_use_grader": False}, QueryClassification=[CLS_EXTERNAL],
                   AnswerDraft=[answer_draft("Open the SecureLink client and connect [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer("What is the latest Microsoft Teams outage guidance?")
    assert r.path == PATH_A and search.call_count == 0 and not r.external_search_triggered and not r.answer.used_external


def test_validator_off_skips_the_grounding_check_and_says_so(make_agent, kb):
    h = make_agent(settings_overrides={"agent_use_validator": False}, QueryClassification=[CLS_INTERNAL], PrivateGrade=[GRADE_OK],
                   AnswerDraft=[answer_draft("Open the SecureLink client and connect [1].")])
    r = h.agent.answer("How do I connect to the company VPN from home?")
    assert r.path == PATH_A and h.llm.count("GroundingCheck") == 0 and not r.answer.abstained
    assert r.validation["passed"] is True and "AGENT_USE_VALIDATOR=false" in r.validation["skipped"]
    assert h.audit.get(r.request_id).validation_result["skipped"]            # the audit record shows nothing was verified


def test_validator_off_delivers_what_the_validator_would_have_blocked(make_agent, kb):
    """The point of the ablation: the same unsupported draft is blocked with the validator on and delivered with it off."""
    kwargs = dict(QueryClassification=[CLS_INTERNAL], PrivateGrade=[GRADE_OK],
                  AnswerDraft=[answer_draft("Open the SecureLink client [1], then reboot the router twice.")])
    on = make_agent(GroundingCheck=[BAD_CHECK], **kwargs).agent.answer("How do I connect to the company VPN from home?")
    off = make_agent(settings_overrides={"agent_use_validator": False}, **kwargs).agent.answer("How do I connect to the company VPN from home?")
    assert on.answer.abstained and on.answer.abstention_reason == "validation_failed"
    assert not off.answer.abstained and "reboot the router" in off.answer.answer


def test_validator_off_does_not_remove_the_abstention_or_external_warning_logic(make_agent, kb):
    """Only the LLM grounding check is skipped: an abstention still passes through unchanged."""
    h = make_agent(settings_overrides={"agent_use_validator": False, "max_rewrite_retries": 0}, QueryClassification=[CLS_INTERNAL],
                   PrivateGrade=[dict(sufficient=False, confidence=0.1, recommended_action="abstain")])
    r = h.agent.answer("What is the cafeteria menu on Friday?")
    assert r.answer.abstained and h.llm.count("GroundingCheck") == 0 and h.search.call_count == 0
