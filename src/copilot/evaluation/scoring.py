"""Per-case scoring: deterministic answer checks and agent-workflow (routing) outcomes."""
from __future__ import annotations

from typing import Any

from copilot.evaluation.results import CaseResult
from copilot.evaluation.schemas import EvalCase
from copilot.evaluation.text import forbidden_hits, normalize, all_keywords_present
from copilot.generation.citations import EXTERNAL_WARNING


def score_answer(case: EvalCase, r: CaseResult, *, answer_threshold: float = 0.6, faithfulness_threshold: float = 0.8) -> dict[str, Any]:
    """Deterministic answer checks (plus judge-derived flags when a judge ran)."""
    norm = normalize(r.answer)
    hits = [all_keywords_present(norm, p.keywords) for p in case.expected_answer_points]
    coverage = sum(hits) / len(hits) if hits else None
    answered = not r.abstained and r.error is None
    leaked = forbidden_hits(r.answer, case.forbidden_keywords)

    if case.expected_abstention:
        correct = r.abstained and r.error is None and not leaked  # an outage is not evidence of correct behaviour
        outcome = "correct_abstention" if correct else ("error" if r.error else "hallucinated_answer")
    else:
        needed = 1.0 if case.expects_conflict_notice else answer_threshold
        if not answered:
            correct = False
        elif coverage is None:  # no keyword labels (live web content): require grounded citations of the expected kind
            correct = r.used_external if case.expects_external else bool(r.cited_documents)
        else:
            correct = coverage >= needed
        correct = correct and not leaked
        outcome = "correct_answer" if correct else ("error" if r.error else ("false_abstention" if not answered else "incorrect_answer"))

    exp_cites = set(case.expected_citations)
    cited = set(r.cited_documents)
    citation_correct: bool | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    if exp_cites and answered:
        citation_precision = len(cited & exp_cites) / len(cited) if cited else 0.0
        citation_recall = len(cited & exp_cites) / len(exp_cites)
        only_expected = cited <= set(case.expected_documents)
        citation_correct = only_expected and (exp_cites <= cited if case.expects_conflict_notice else bool(cited & exp_cites))

    faithful = relevant = None
    if r.judge is not None and answered:
        faithful = (r.judge.faithfulness is not None) and r.judge.faithfulness >= faithfulness_threshold
        relevant = r.judge.relevance
    return {
        "point_coverage": coverage, "points_met": sum(hits), "points_total": len(hits),
        "correct": correct, "outcome": outcome, "leaked_keywords": leaked,
        "citation_correct": citation_correct, "citation_precision": citation_precision, "citation_recall": citation_recall,
        "external_label_ok": (EXTERNAL_WARNING in r.answer) if r.used_external else None,
        "faithful": faithful, "unsupported_claims": (not faithful) if faithful is not None else None, "answer_relevance": relevant,
    }


def routing_outcomes(case: EvalCase, r: CaseResult) -> dict[str, Any]:
    """Workflow-level outcomes. Only the agent (any variant) has a route, so baseline results return {}."""
    if r.mode == "baseline":
        return {}
    retr = r.path.index("private_retriever") if "private_retriever" in r.path else None
    ext = r.path.index("external_search") if "external_search" in r.path else None
    if case.expected_route == "out_of_scope":
        private_first = None
    else:
        private_first = retr is not None and (ext is None or retr < ext)
    attempted = r.retry_count > 0
    recoverable = not case.expects_external and not case.expected_abstention
    expected_sufficient = case.expects_sufficient_private_evidence
    return {
        "private_first_ok": private_first,
        "unnecessary_external": bool(r.external_triggered and not case.expects_external),
        "external_fallback_ok": (r.external_triggered and r.used_external) if (case.expects_external and case.external_search_allowed) else None,
        "rewrite_attempted": attempted,
        "rewrite_success": (r.grade_sufficient is True) if (attempted and recoverable) else None,
        "grading_correct": (r.grade_sufficient == expected_sufficient) if (expected_sufficient is not None and r.grade_sufficient is not None) else None,
        "validation_failed": r.abstention_reason == "validation_failed",
    }


def score_case(case: EvalCase, r: CaseResult, *, answer_threshold: float = 0.6, faithfulness_threshold: float = 0.8) -> dict[str, Any]:
    """Merge answer and routing scores into the dict stored on ``CaseResult.scores``."""
    return {**score_answer(case, r, answer_threshold=answer_threshold, faithfulness_threshold=faithfulness_threshold), **routing_outcomes(case, r)}
