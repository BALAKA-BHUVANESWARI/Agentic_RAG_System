"""Experiment A (docs/POST_DAY3_EXPERIMENTS.md): the external-information separation is decided in code for external-only answers.

The flag EXTERNAL_LABEL_CHECK_IN_CODE is off by default. These tests use a scripted LLM, so they show the wiring and
the guard rails; they are NOT evidence that the change improves answers (that needs the planned runs).
"""
from __future__ import annotations

from fakes import FakeSearch

from copilot.generation.citations import EXTERNAL_WARNING
from copilot.generation.models import Citation, GeneratedAnswer
from copilot.graph.nodes.final_answer_validator import external_separation_by_construction
from workflow.test_scenarios import CLS_BOTH, CLS_EXTERNAL, EXT_ACCEPT, GRADE_BAD, GRADE_OK, TEAMS_RESULT, answer_draft

NOT_SEPARATED = dict(grounded=True, external_clearly_separated=False, reasoning="no company guidance to separate from")
EXTERNAL_ONLY = dict(
    QueryClassification=[CLS_EXTERNAL], PrivateGrade=[GRADE_BAD], ExternalGrade=[EXT_ACCEPT],
    AnswerDraft=[answer_draft("Check the service health page for current Teams incidents [1].")],
)
QUESTION = "What is the latest Microsoft Teams outage guidance?"


def test_flag_is_off_by_default_and_recorded_in_the_run_snapshot(make_settings):
    from copilot.evaluation.system import settings_snapshot

    assert make_settings().external_label_check_in_code is False
    assert settings_snapshot(make_settings())["external_label_check_in_code"] is False
    assert settings_snapshot(make_settings(external_label_check_in_code=True))["external_label_check_in_code"] is True


def test_control_an_llm_separation_veto_on_an_external_only_answer_abstains_as_in_fresh_1(make_agent, kb):
    """Flag OFF reproduces the fresh-1 F26-F28 behaviour: a labelled, cited external answer is rejected twice."""
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides={"max_rewrite_retries": 0},
                   GroundingCheck=[NOT_SEPARATED], **EXTERNAL_ONLY)
    r = h.agent.answer(QUESTION, "employee")
    assert r.path.count("answer_generator") == 2 and r.answer.abstained and r.answer.abstention_reason == "validation_failed"
    assert r.validation["rejected_answer"].startswith(EXTERNAL_WARNING)  # the rejected draft was labelled and cited


def test_treatment_external_only_answer_passes_without_asking_the_llm_to_judge_separation(make_agent, kb):
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides={"max_rewrite_retries": 0, "external_label_check_in_code": True},
                   GroundingCheck=[NOT_SEPARATED], **EXTERNAL_ONLY)
    r = h.agent.answer(QUESTION, "employee")
    assert r.path.count("answer_generator") == 1 and not r.answer.abstained
    assert r.answer.rendered.startswith(EXTERNAL_WARNING) and r.answer.citations[0].url == TEAMS_RESULT.url
    assert r.validation["passed"] is True and r.validation["external_separation_decided_in_code"] is True


def test_treatment_still_rejects_an_unsupported_claim_in_an_external_answer(make_agent, kb):
    """The flag only removes the separation veto; the grounding check still runs and still fails closed."""
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides={"max_rewrite_retries": 0, "external_label_check_in_code": True},
                   GroundingCheck=[dict(grounded=False, unsupported_claims=["Microsoft will fix it in 5 minutes"],
                                        external_clearly_separated=True)], **EXTERNAL_ONLY)
    r = h.agent.answer(QUESTION, "employee")
    assert r.answer.abstained and r.answer.abstention_reason == "validation_failed"


def test_treatment_mixed_private_and_external_answers_still_use_the_llm_judgement(make_agent, kb):
    """Scenario C blends two kinds of source, so the flag does not cover it: the LLM veto still triggers a revision."""
    h = make_agent(search=FakeSearch([TEAMS_RESULT]), settings_overrides={"external_label_check_in_code": True},
                   QueryClassification=[CLS_BOTH], PrivateGrade=[dict(GRADE_OK, needs_external_supplement=True)],
                   ExternalGrade=[EXT_ACCEPT],
                   AnswerDraft=[answer_draft("Company guidance: report issues to the Service Desk [1]. Public vendor status page: check service health [2].")],
                   GroundingCheck=[NOT_SEPARATED, dict(grounded=True, external_clearly_separated=True)])
    r = h.agent.answer("How should I use Teams at work and is Teams having known issues right now?", "employee")
    assert [c.kind for c in r.answer.citations] == ["private", "external"]
    assert r.path.count("answer_generator") == 2 and r.validation["external_separation_decided_in_code"] is False
    assert not r.answer.abstained


def _answer(kinds: list[str], rendered_prefix: str = EXTERNAL_WARNING, url: str | None = "https://learn.microsoft.com/x") -> GeneratedAnswer:
    cites = [Citation(ref=i + 1, kind=k, label="L", location="loc", url=url if k == "external" else None) for i, k in enumerate(kinds)]
    return GeneratedAnswer(answer="x [1]", rendered=f"{rendered_prefix}\n\nx [1]", citations=cites, used_external="external" in kinds)


def test_separation_by_construction_truth_table():
    assert external_separation_by_construction(_answer(["external"])) is True
    assert external_separation_by_construction(_answer(["external", "external"])) is True
    assert external_separation_by_construction(_answer(["private", "external"])) is False      # mixed: not covered
    assert external_separation_by_construction(_answer(["private"])) is False                  # not an external answer
    assert external_separation_by_construction(_answer(["external"], rendered_prefix="Intro")) is False  # label line not first
    assert external_separation_by_construction(_answer(["external"], url=None)) is False       # a citation without a URL
    assert external_separation_by_construction(_answer([])) is False
