"""Retrieval metrics (hand-computed), answer scoring, routing outcomes and aggregation."""
from __future__ import annotations

import pytest

from copilot.evaluation import aggregate as agg
from copilot.evaluation.results import CaseResult, JudgeScores, RetrievalCaseResult, RetrievedItem
from copilot.evaluation.retrieval_metrics import chunk_is_relevant, ndcg_at_k, reciprocal_rank, score_retrieval
from copilot.evaluation.schemas import AnswerPoint, EvalCase
from copilot.evaluation.scoring import routing_outcomes, score_case
from copilot.generation.citations import EXTERNAL_WARNING


def case(**kw) -> EvalCase:
    base = dict(id="T1", category="direct_kb", question="q", expected_route="private_only", expected_documents=["A.txt"], expected_citations=["A.txt"],
                expected_abstention=False, expected_answer_points=[AnswerPoint(text="x", keywords=["alpha"]), AnswerPoint(text="y", keywords=["beta", "gamma"])])
    base.update(kw)
    return EvalCase(**base)


def result(**kw) -> CaseResult:
    base = dict(case_id="T1", category="direct_kb", mode="agentic", question="q", answer="alpha and beta gamma", cited_documents=["A.txt"])
    base.update(kw)
    return CaseResult(**base)


def item(doc, text="", cid=None):
    return RetrievedItem(document_name=doc, chunk_id=cid or f"{doc}:{text[:5]}", text=text)


# ------------------------------------------------------------------------------ retrieval
def test_retrieval_metrics_hand_computed():
    c = case(expected_answer_points=[AnswerPoint(text="x", keywords=["alpha"])])
    items = [item("B.txt", "alpha"), item("A.txt", "has alpha here"), item("A.txt", "unrelated"), item("B.txt", "zzz")]
    assert [chunk_is_relevant(c, i) for i in items] == [False, True, False, False]  # right doc AND evidence keyword
    s = score_retrieval(c, items, [1, 3, 4])
    assert s["chunk_mrr"] == 0.5 and s["doc_mrr"] == 0.5
    assert s["doc_recall@1"] == 0.0 and s["doc_recall@3"] == 1.0
    assert s["doc_precision@3"] == 0.5          # docs in top3 = {B, A}, relevant = {A}
    assert s["chunk_precision@3"] == pytest.approx(1 / 3) and s["chunk_precision@1"] == 0.0
    assert s["chunk_recall@3"] == 1.0           # pool-based: its only relevant chunk is within top 3
    assert s["chunk_ndcg@3"] == pytest.approx(0.6309297535714574)  # 1/log2(3) over an ideal of 1/log2(2)
    assert s["chunk_ndcg@1"] == 0.0


def test_ndcg_and_reciprocal_rank_edges():
    assert ndcg_at_k([True, True, False], 3, 2) == pytest.approx(1.0)
    assert ndcg_at_k([False, True], 2, 1) == pytest.approx(0.6309297535714574)
    assert ndcg_at_k([False, False], 2, 0) is None
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3) and reciprocal_rank([False]) == 0.0


def test_retrieval_with_nothing_relevant_scores_zero_and_no_expected_docs_scores_nothing():
    c = case()
    s = score_retrieval(c, [item("B.txt", "nothing")], [1, 5])
    assert s["doc_recall@5"] == 0.0 and s["chunk_ndcg@5"] == 0.0 and s["chunk_mrr"] == 0.0
    assert score_retrieval(case(expected_route="abstain", expected_documents=[], expected_citations=[], expected_abstention=True), [item("B.txt")], [5]) == {}


# ------------------------------------------------------------------------------ answers
def test_correct_answer_needs_threshold_coverage_and_no_leak():
    c = case()
    assert score_case(c, result())["correct"] is True
    half = score_case(c, result(answer="only alpha"))
    assert half["point_coverage"] == 0.5 and half["correct"] is False and half["outcome"] == "incorrect_answer"
    assert score_case(c, result(answer="alpha beta gamma"), answer_threshold=0.6)["points_met"] == 2
    leaky = case(forbidden_keywords=["secret tool"])
    assert score_case(leaky, result(answer="alpha beta gamma uses the secret tool"))["correct"] is False


def test_abstention_outcomes():
    c = case(expected_route="abstain", expected_documents=[], expected_citations=[], expected_answer_points=[], expected_abstention=True)
    assert score_case(c, result(abstained=True, answer="no info", cited_documents=[]))["outcome"] == "correct_abstention"
    assert score_case(c, result(answer="made up", cited_documents=[]))["outcome"] == "hallucinated_answer"
    ans = case()
    assert score_case(ans, result(abstained=True, answer="no info"))["outcome"] == "false_abstention"
    assert score_case(ans, result(error="boom"))["correct"] is False


def test_abstain_case_with_leak_is_not_a_correct_abstention():
    c = case(expected_route="abstain", expected_documents=[], expected_citations=[], expected_answer_points=[], expected_abstention=True, forbidden_keywords=["paging tool"])
    assert score_case(c, result(abstained=True, answer="Use the paging tool", cited_documents=[]))["correct"] is False


def test_conflict_cases_need_every_version_and_both_citations():
    c = case(expected_documents=["A.txt", "B.txt"], expected_citations=["A.txt", "B.txt"], expects_conflict_notice=True)
    one = score_case(c, result(answer="alpha only", cited_documents=["A.txt"]))
    both = score_case(c, result(answer="alpha beta gamma", cited_documents=["A.txt", "B.txt"]))
    assert one["correct"] is False and both["correct"] is True and both["citation_correct"] is True and one["citation_correct"] is False


def test_citation_scoring():
    c = case()
    ok = score_case(c, result())
    assert ok["citation_correct"] is True and ok["citation_precision"] == 1.0 and ok["citation_recall"] == 1.0
    extra = score_case(c, result(cited_documents=["A.txt", "Z.txt"]))
    assert extra["citation_correct"] is False and extra["citation_precision"] == 0.5
    assert score_case(c, result(cited_documents=[]))["citation_correct"] is False
    assert score_case(c, result(abstained=True))["citation_correct"] is None


def test_external_answers_are_checked_for_the_warning_and_keyword_free_cases_need_external_citations():
    ext = case(expected_route="private_then_external", expected_documents=[], expected_citations=[], expected_answer_points=[])
    labelled = score_case(ext, result(answer=f"{EXTERNAL_WARNING}\n\nSee page [1].", used_external=True, cited_documents=[]))
    assert labelled["correct"] is True and labelled["external_label_ok"] is True
    unlabelled = score_case(ext, result(answer="See page [1].", used_external=True, cited_documents=[]))
    assert unlabelled["external_label_ok"] is False
    assert score_case(ext, result(answer="private only", cited_documents=["A.txt"]))["correct"] is False


def test_judge_flags():
    s = score_case(case(), result(judge=JudgeScores(faithfulness=0.5, relevance=0.9)), faithfulness_threshold=0.8)
    assert s["faithful"] is False and s["unsupported_claims"] is True and s["answer_relevance"] == 0.9
    assert score_case(case(), result())["faithful"] is None


# ------------------------------------------------------------------------------ routing
def test_routing_private_first_and_unnecessary_external():
    c = case()
    good = routing_outcomes(c, result(path=["query_classifier", "private_retriever", "private_evidence_grader"]))
    assert good["private_first_ok"] is True and good["unnecessary_external"] is False
    bad = routing_outcomes(c, result(path=["query_classifier", "external_search"], external_triggered=True))
    assert bad["private_first_ok"] is False and bad["unnecessary_external"] is True
    oos = case(expected_route="out_of_scope", expected_documents=[], expected_citations=[], expected_answer_points=[], expected_abstention=True)
    assert routing_outcomes(oos, result(path=["query_classifier"]))["private_first_ok"] is None
    assert routing_outcomes(c, result(mode="baseline")) == {}


def test_routing_external_fallback_rewrite_and_grading():
    ext = case(expected_route="private_then_external", expected_documents=[], expected_citations=[], expected_answer_points=[])
    r = routing_outcomes(ext, result(external_triggered=True, used_external=True, grade_sufficient=False, path=["private_retriever", "external_search"]))
    assert r["external_fallback_ok"] is True and r["grading_correct"] is True and r["unnecessary_external"] is False
    forbidden = routing_outcomes(case(expected_route="private_then_external", expected_documents=[], expected_citations=[], expected_answer_points=[], external_search_allowed=False), result())
    assert forbidden["external_fallback_ok"] is None
    rw = routing_outcomes(case(), result(retry_count=1, grade_sufficient=True))
    assert rw["rewrite_attempted"] is True and rw["rewrite_success"] is True and rw["grading_correct"] is True
    assert routing_outcomes(case(), result(retry_count=0))["rewrite_success"] is None
    assert routing_outcomes(case(), result(abstention_reason="validation_failed", abstained=True))["validation_failed"] is True


# ------------------------------------------------------------------------------ aggregation
def test_percentile_mean_rate_and_cost(make_settings):
    assert agg.percentile(list(range(1, 21)), 95) == 19 and agg.percentile([5.0], 95) == 5.0 and agg.percentile([], 95) is None
    assert agg.mean([1, None, 3]) == 2 and agg.mean([None]) is None
    assert agg.rate(1, 4) == {"value": 0.25, "num": 1, "den": 4} and agg.rate(0, 0)["value"] is None
    assert agg.estimate_cost(1_000_000, 1_000_000, 10, make_settings()) == pytest.approx(0.15 + 0.60 + 0.08)


def _scored(c, r, settings):
    r.scores = score_case(c, r)
    return r


def test_summarize_mode_and_compare(make_settings):
    s = make_settings()
    cases = [
        case(id="A1"),
        case(id="A2", category="paraphrased"),
        case(id="U1", category="unanswerable", expected_route="abstain", expected_documents=[], expected_citations=[], expected_answer_points=[], expected_abstention=True),
        case(id="E1", category="current_external", expected_route="private_then_external", expected_documents=[], expected_citations=[], expected_answer_points=[]),
    ]
    path = ["query_classifier", "private_retriever", "private_evidence_grader"]
    agentic = [
        _scored(cases[0], result(case_id="A1", path=path, grade_sufficient=True, grade_confidence=0.9, latency_ms=1000, prompt_tokens=100, completion_tokens=10), s),
        _scored(cases[1], result(case_id="A2", answer="alpha", path=path, grade_sufficient=True, grade_confidence=0.7, latency_ms=3000), s),
        _scored(cases[2], result(case_id="U1", abstained=True, answer="none", cited_documents=[], path=path, grade_sufficient=False, grade_confidence=0.9, external_triggered=True, tavily_calls=1, latency_ms=2000), s),
        _scored(cases[3], result(case_id="E1", answer=EXTERNAL_WARNING + " x", used_external=True, cited_documents=[], external_triggered=True, tavily_calls=1,
                                  path=path + ["external_search"], grade_sufficient=False, grade_confidence=0.8, latency_ms=4000), s),
    ]
    baseline = [
        _scored(cases[0], result(case_id="A1", mode="baseline", latency_ms=500), s),
        _scored(cases[1], result(case_id="A2", mode="baseline", answer="alpha beta gamma", latency_ms=500), s),
        _scored(cases[2], result(case_id="U1", mode="baseline", answer="made up", cited_documents=[], latency_ms=500), s),
        _scored(cases[3], result(case_id="E1", mode="baseline", answer="no web", cited_documents=[], latency_ms=500), s),
    ]
    a, b = agg.summarize_mode(cases, agentic, s), agg.summarize_mode(cases, baseline, s)

    assert a["answer"]["overall_correct_rate"] == {"value": 0.75, "num": 3, "den": 4}   # A1, U1, E1 correct; A2 incomplete
    assert a["answer"]["correct_abstention_rate"]["value"] == 1.0 and b["answer"]["hallucinated_answer_rate"]["value"] == 1.0
    assert a["routing"]["unnecessary_external_search_rate"] == {"value": pytest.approx(1 / 3), "num": 1, "den": 3}   # U1 searched but should not have
    assert a["routing"]["external_fallback_accuracy"]["value"] == 1.0 and a["routing"]["private_first_routing_accuracy"]["value"] == 1.0
    assert a["routing"]["evidence_grading_accuracy"]["den"] == 4 and a["routing"]["tavily_calls_total"] == 2 and b["routing"] == {}
    assert a["system"]["latency_avg_ms"] == 2500 and a["system"]["latency_p95_ms"] == 4000
    assert a["by_category"]["paraphrased"]["value"] == 0.0 and a["system"]["total_tokens"] == 110

    cmp = agg.compare_modes(b, a)
    assert "Overall correct rate" in cmp["agentic_better"] and "Average latency (ms)" in cmp["baseline_better"]
    assert any("single-run" in n for n in cmp["findings"])


def test_compare_modes_reports_ties_and_missing_values():
    base = {"answer": {"overall_correct_rate": {"value": 0.80}}, "system": {}}
    new = {"answer": {"overall_correct_rate": {"value": 0.81}}, "system": {}}
    cmp = agg.compare_modes(base, new)
    row = next(r for r in cmp["rows"] if r["metric"] == "Overall correct rate")
    assert row["verdict"] == "tie" and cmp["agentic_better"] == []
    assert next(r for r in cmp["rows"] if r["metric"] == "Total tokens")["verdict"] == "n/a"


def test_summarize_retrieval_means_only_scored_cases():
    rr = [RetrievalCaseResult(case_id="1", category="x", scores={"doc_recall@5": 1.0, "chunk_mrr": 0.5}),
          RetrievalCaseResult(case_id="2", category="x", scores={"doc_recall@5": 0.0, "chunk_mrr": 1.0}),
          RetrievalCaseResult(case_id="3", category="x", scores={})]
    s = agg.summarize_retrieval(rr, [5])
    assert s["n_cases"] == 2 and s["metrics"] == {"chunk_mrr": 0.75, "doc_recall@5": 0.5}
