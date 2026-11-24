"""Experiment B (docs/POST_DAY3_EXPERIMENTS.md): keep the grader's supporting private chunks when the external half is not established.

The flag PARTIAL_PRIVATE_EVIDENCE is off by default. These tests use a scripted LLM, so they show the wiring and the guard rails
(the E05 pattern from test-1); they are NOT evidence that the change improves answers.
"""
from __future__ import annotations

import pytest
from fakes import FakeSearch

from copilot.graph.nodes.answer_generator import partial_private_applies
from workflow.test_scenarios import (
    CLS_BOTH, CLS_EXTERNAL, EXT_ACCEPT, EXT_REJECT, GRADE_OK, OK_CHECK, PATH_A, TEAMS_RESULT, answer_draft,
)

QUESTION = "How long are Teams meeting recordings kept at our company, and is Microsoft reporting any Teams outage today?"
# The E05 grade: supporting chunk found, high confidence, but "insufficient" because the question also needs live public data.
GRADE_PARTIAL = dict(sufficient=False, confidence=0.95, supporting_refs=[1], needs_external_supplement=True,
                     missing_information=["current outage status"], recommended_action="external_search")
PARTIAL_DRAFT = dict(can_answer=True, answer="Company guidance: recordings are stored for a limited time [1].",
                     missing_information="whether Microsoft reports an outage today")
ON = {"max_rewrite_retries": 0, "partial_private_evidence": True}
OFF = {"max_rewrite_retries": 0}


def test_flag_is_off_by_default_and_recorded_in_the_run_snapshot(make_settings):
    from copilot.evaluation.system import settings_snapshot

    assert make_settings().partial_private_evidence is False
    assert settings_snapshot(make_settings())["partial_private_evidence"] is False
    assert settings_snapshot(make_settings(partial_private_evidence=True))["partial_private_evidence"] is True


def test_control_the_supported_private_half_is_dropped_and_the_answer_abstains_as_in_test_1_e05(make_agent, kb):
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides=OFF, QueryClassification=[CLS_BOTH],
                   PrivateGrade=[GRADE_PARTIAL], ExternalGrade=[EXT_REJECT])
    r = h.agent.answer(QUESTION, "employee")
    assert r.answer.abstained and r.answer.abstention_reason == "insufficient_evidence"
    assert h.llm.count("AnswerDraft") == 0 and r.tavily_calls >= 1


def test_treatment_answers_the_supported_private_half_when_no_external_source_is_accepted(make_agent, kb):
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides=ON, QueryClassification=[CLS_BOTH],
                   PrivateGrade=[GRADE_PARTIAL], ExternalGrade=[EXT_REJECT], AnswerDraft=[PARTIAL_DRAFT], GroundingCheck=[OK_CHECK])
    r = h.agent.answer(QUESTION, "employee")
    assert not r.answer.abstained and [c.kind for c in r.answer.citations] == ["private"]
    prompt = h.llm.prompt("AnswerDraft", 0)  # the prompt also holds the system prompt, so look at the evidence tags only
    assert prompt.count('<evidence id="') == 1 and '<evidence id="1" type="private"' in prompt and 'type="external" source' not in prompt
    ev = [e for e in r.trace if e.node == "answer_generator"][-1]
    assert ev.data["partial_private_used"] is True and ev.data["private_sources"] == 1 and ev.data["external_sources"] == 0


def test_treatment_private_chunk_comes_first_and_the_external_page_is_labelled_when_both_are_available(make_agent, kb):
    draft = answer_draft("Company guidance: recordings are kept for a limited time [1]. Public vendor page: check service health [2].")
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides=ON, QueryClassification=[CLS_BOTH],
                   PrivateGrade=[GRADE_PARTIAL], ExternalGrade=[EXT_ACCEPT], AnswerDraft=[draft], GroundingCheck=[OK_CHECK])
    r = h.agent.answer(QUESTION, "employee")
    assert [c.kind for c in r.answer.citations] == ["private", "external"] and not r.answer.abstained
    assert r.answer.rendered.startswith("External information:")


@pytest.mark.parametrize("cls,grade,why", [
    (CLS_EXTERNAL, GRADE_PARTIAL, "classifier intent is not 'both'"),
    (CLS_BOTH, dict(GRADE_PARTIAL, confidence=0.3), "confidence below the private threshold"),
    (CLS_BOTH, dict(GRADE_PARTIAL, needs_external_supplement=False), "grader sees no public need"),
    (CLS_BOTH, dict(GRADE_PARTIAL, contradiction_detected=True), "contradicting chunks are not partially used"),
    (CLS_BOTH, dict(GRADE_PARTIAL, supporting_refs=[]), "no supporting chunk named"),
])
def test_treatment_guard_rails_keep_the_old_behaviour(make_agent, kb, cls, grade, why):
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides=ON, QueryClassification=[cls],
                   PrivateGrade=[grade], ExternalGrade=[EXT_REJECT], AnswerDraft=[PARTIAL_DRAFT], GroundingCheck=[OK_CHECK])
    r = h.agent.answer(QUESTION, "employee")
    assert r.answer.abstained, why
    assert h.llm.count("AnswerDraft") == 0, why


def test_treatment_changes_nothing_when_the_private_evidence_is_already_sufficient(make_agent, kb):
    h = make_agent(settings_overrides=ON, QueryClassification=[dict(intent="internal_it", confidence=0.9)], PrivateGrade=[GRADE_OK],
                   AnswerDraft=[answer_draft("Open the SecureLink client and connect [1].")], GroundingCheck=[OK_CHECK])
    r = h.agent.answer("How do I connect to the company VPN from home?", "employee")
    assert r.path == PATH_A and [c.kind for c in r.answer.citations] == ["private"]
    ev = [e for e in r.trace if e.node == "answer_generator"][-1]
    assert ev.data["partial_private_used"] is False


def test_partial_private_applies_requires_the_flag(make_settings):
    state = {"classification": {"intent": "both"},
             "private_grade": {"needs_external_supplement": True, "confidence": 0.95, "threshold": 0.6, "supporting_chunk_ids": ["c1"]}}
    assert partial_private_applies(state, make_settings(partial_private_evidence=True)) is True
    assert partial_private_applies(state, make_settings()) is False
