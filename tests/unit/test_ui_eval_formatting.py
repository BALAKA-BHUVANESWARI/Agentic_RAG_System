"""Evaluation-dashboard formatting: pure functions over a run summary (no Streamlit, no network)."""
from __future__ import annotations

import copy

import pytest

from copilot.ui import eval_formatting as ef


def rate(num: int, den: int) -> dict:
    return {"value": (num / den) if den else None, "num": num, "den": den}


def _system(n: int, latency: float, tokens: int, cost: float) -> dict:
    return {
        "n_cases": n, "latency_avg_ms": latency, "latency_p95_ms": latency * 2, "error_rate": rate(0, n), "prompt_tokens": tokens - 100,
        "completion_tokens": 100, "total_tokens": tokens, "llm_calls": n * 2, "tavily_calls": 3, "estimated_cost_usd": cost,
    }


def _answer(correct: int, n: int = 24) -> dict:
    return {
        "overall_correct_rate": rate(correct, n), "answer_correct_rate": rate(correct - 9, 15), "answer_point_coverage_mean": 0.5,
        "false_abstention_rate": rate(15 - (correct - 9), 15), "correct_abstention_rate": rate(9, 9), "hallucinated_answer_rate": rate(0, 9),
        "citation_correct_rate": rate(6, 7), "conflict_handling_rate": rate(2, 3), "external_label_rate": rate(0, 0),
        "restricted_content_leaks": 0, "faithfulness_rate": rate(7, 7), "unsupported_claim_rate": rate(0, 7), "answer_relevance_mean": 1.0,
    }


def make_summary(n: int = 24, *, status: str = "completed", agentic_correct: int = 14, baseline_correct: int = 16, latency: float = 2000.0) -> dict:
    agentic_better = ["Overall correct rate"] if agentic_correct > baseline_correct else []
    baseline_better = ["Overall correct rate"] if baseline_correct > agentic_correct else []
    return {
        "run_id": "20261117T101736Z-420ac6", "status": status, "error": None,
        "options": {"mode": "both", "per_category": 3, "limit": None, "categories": None, "use_judge": True},
        "dataset": {"num_cases": n, "category_counts": {"direct_kb": 3, "conflicting": 3}},
        "started_at": "2026-11-17T10:17:36+00:00", "finished_at": "2026-11-17T10:33:15+00:00",
        "settings": {"llm_model": "openai/gpt-oss-20b", "reasoning_effort": "low"},
        "progress": {"units_total": n * 2, "units_recorded": n * 2, "units_ok": n * 2},
        "retrieval": {"n_cases": 12, "ks": [3, 5], "metrics": {
            "doc_mrr": 1.0, "chunk_mrr": 0.9167, "doc_recall@3": 1.0, "doc_precision@3": 0.9583, "doc_ndcg@3": 0.98,
            "doc_recall@5": 1.0, "doc_precision@5": 0.8333, "doc_ndcg@5": 0.94,
            "chunk_recall@3": 0.84, "chunk_precision@3": 0.47, "chunk_ndcg@3": 0.84, "chunk_recall@5": 0.91, "chunk_precision@5": 0.32, "chunk_ndcg@5": 0.87}},
        "modes": {
            "baseline": {"answer": _answer(baseline_correct, n), "routing": {}, "system": _system(n, latency, 21665, 0.0043),
                         "by_category": {"direct_kb": rate(2, 3), "conflicting": rate(3, 3)}},
            "agentic": {
                "answer": _answer(agentic_correct, n),
                "routing": {
                    "private_first_routing_accuracy": rate(19, 21), "correct_private_only_answer_rate": rate(5, 12), "unnecessary_external_search_rate": rate(1, 21),
                    "external_fallback_accuracy": rate(0, 3), "query_rewrite_success_rate": rate(1, 1), "evidence_grading_accuracy": rate(19, 19),
                    "correct_abstention_rate": rate(9, 9), "final_validation_failure_rate": rate(2, 24),
                    "grader_confidence_calibration": [{"range": "0.0-0.5", "n": 8, "accuracy": 1.0}, {"range": "0.9-1.0", "n": 9, "accuracy": 1.0}],
                    "external_searches_total": 4, "tavily_calls_total": 6, "cases_with_rewrite": 9},
                "system": _system(n, latency * 9, 104668, 0.0712),
                "by_category": {"direct_kb": rate(3, 3), "conflicting": rate(1, 3)}},
        },
        "comparison": {
            "rows": [
                {"metric": "Overall correct rate", "baseline": baseline_correct / n, "agentic": agentic_correct / n, "verdict": "baseline better"},
                {"metric": "Average latency (ms)", "baseline": latency, "agentic": latency * 9, "verdict": "baseline better"},
                {"metric": "Total tokens", "baseline": 21665, "agentic": 104668, "verdict": "baseline better"},
                {"metric": "Estimated cost (USD)", "baseline": 0.0043, "agentic": 0.0712, "verdict": "baseline better"},
            ],
            "findings": ["Baseline is better on 1 metric(s): Overall correct rate."], "agentic_better": agentic_better, "baseline_better": baseline_better, "tied": []},
        "ingestion": {"attempts": rate(11, 15), "documents": rate(11, 11)},
        "judge": {"enabled": True, "judged_answers": 12, "tokens": 12000, "calls": 12},
        "failed_cases": [{"case_id": "C01", "category": "conflicting", "mode": "agentic", "question": "What is the minimum password length?", "expected_route": "private_only",
                          "outcome": "false_abstention", "abstention_reason": "validation_failed", "error": None, "points_met": 0, "points_total": 2,
                          "cited_documents": [], "path": ["query_classifier", "private_retriever"], "answer": "I couldn't verify...", "request_id": "r1"}],
        "notes": ["All results are from a single run."],
    }


# ------------------------------------------------------------------------------ number formatting
def test_rates_and_values_are_formatted_for_reading():
    assert ef.pct(0.6667) == "67%" and ef.pct(None) == "n/a"
    assert ef.rate_text(rate(16, 24)) == "67% (16/24)"
    assert ef.rate_text(rate(0, 0)) == "n/a" and ef.rate_text(None) == "n/a"          # no denominator: not applicable, never "0%"
    assert ef.metric_value_text("Average latency (ms)", 26754.4) == "26.8 s" and ef.metric_value_text("Total tokens", 104668) == "104,668"
    assert ef.metric_value_text("Estimated cost (USD)", 0.07121) == "$0.0712" and ef.metric_value_text("Overall correct rate", 0.5833) == "58%"
    assert ef.metric_value_text("Overall correct rate", None) == "n/a"
    assert ef.duration_text("2026-11-17T10:17:36+00:00", "2026-11-17T10:33:15+00:00") == "15 min 39 s"
    assert ef.duration_text("2026-11-17T10:17:36+00:00", "2026-11-17T10:17:50+00:00") == "14 s" and ef.duration_text(None, None) == "-"


# ------------------------------------------------------------------------------ run list and status
@pytest.mark.parametrize("status,level", [("running", "info"), ("cancelling", "warning"), ("completed", "success"), ("cancelled", "warning"),
                                          ("stopped", "error"), ("failed", "error"), ("interrupted", "warning"), ("weird", "info")])
def test_status_banner_levels(status, level):
    got, text = ef.status_banner({"status": status, "done": 3, "total": 16, "message": "R01 [agentic]", "error": "quota"})
    assert got == level and text


def test_stopped_banner_explains_quota_and_resume():
    _, text = ef.status_banner({"status": "stopped", "done": 1, "total": 48, "error": "Resume later: --resume x"})
    assert "quota" in text and "Resume" in text and "1/48" in text
    assert "--resume" not in text and text.count("quota") == 1          # no CLI hint in the browser, no repeated sentence


def test_run_selection_helpers():
    runs = [{"run_id": "a", "status": "running", "done": 2, "total": 8}, {"run_id": "b", "status": "stopped"}, {"run_id": "c", "status": "completed"},
            {"run_id": "d", "status": "cancelled"}, {"run_id": "e", "status": "completed"}]
    assert ef.active_run(runs)["run_id"] == "a" and ef.active_run(runs[1:]) is None
    assert [r["run_id"] for r in ef.resumable_runs(runs)] == ["b", "d"]
    assert ef.latest_completed(runs)["run_id"] == "c"
    assert ef.progress_fraction(runs[0]) == 0.25 and ef.progress_fraction({"done": 5, "total": 0}) == 0.0 and ef.progress_fraction({"done": 9, "total": 4}) == 1.0
    assert ef.run_label({"run_id": "a", "status": "running", "mode": "both", "num_cases": 8}) == "a · running · both · 8 cases"
    assert ef.run_rows(runs)[0]["progress"] == "2/8"


# ------------------------------------------------------------------------------ pre-run estimate
def test_selected_question_count():
    assert ef.selected_question_count(None, None, None) is None                       # whole dataset: size unknown to the UI
    assert ef.selected_question_count(3, None, None) == 24                            # 3 x 8 categories (upper bound)
    assert ef.selected_question_count(3, None, ["direct_kb", "conflicting"]) == 6
    assert ef.selected_question_count(3, 5, None) == 5 and ef.selected_question_count(None, 7, None) == 7


def test_token_estimate_scales_from_a_measured_run():
    s = make_summary()
    both = ef.estimate_tokens(s, 8, "both", use_judge=False)
    assert both == int(21665 / 24 * 8 + 104668 / 24 * 8)
    assert ef.estimate_tokens(s, 8, "agentic", use_judge=False) == int(104668 / 24 * 8)
    assert ef.estimate_tokens(s, 8, "agentic", use_judge=True) == int(104668 / 24 * 8 + 12000 / 12 * 8)
    assert ef.estimate_tokens(None, 8, "both", False) is None and ef.estimate_tokens(s, None, "both", False) is None
    one_sided = copy.deepcopy(s)
    del one_sided["modes"]["agentic"]
    assert ef.estimate_tokens(one_sided, 8, "both", False) is None                    # cannot estimate a system never measured


# ------------------------------------------------------------------------------ caveats and verdict
def test_small_sample_model_and_judge_caveats_are_always_shown():
    texts = " ".join(t for _, t in ef.caveats(make_summary()))
    assert "24 labelled questions" in texts and "noise" in texts
    assert "openai/gpt-oss-20b" in texts and "not comparable" in texts
    assert "LLM judge" in texts


def test_partial_error_and_one_sided_runs_are_flagged():
    partial = make_summary(status="cancelled")
    partial["progress"] = {"units_total": 48, "units_recorded": 9, "units_ok": 9}
    texts = " ".join(t for _, t in ef.caveats(partial))
    assert "cancelled" in texts and "39 question run(s) did not finish" in texts
    one = make_summary()
    del one["modes"]["agentic"]
    assert "Only the Baseline RAG has results" in " ".join(t for _, t in ef.caveats(one))


def test_slow_latency_adds_the_rate_limit_note_only_when_slow():
    assert not any("rate limits" in t for _, t in ef.caveats(make_summary(latency=1000)))
    assert any("rate limits" in t for _, t in ef.caveats(make_summary(latency=5000)))        # agentic = 9x = 45 s


def test_verdict_never_claims_agentic_wins_on_a_small_sample():
    level, text = ef.comparison_verdict(make_summary(n=24, agentic_correct=20, baseline_correct=10))
    assert level == "warning" and "Not enough evidence" in text and "Agentic wins 1 metric(s)" in text


def test_verdict_on_a_large_sample_follows_the_data():
    assert ef.comparison_verdict(make_summary(n=60, agentic_correct=40, baseline_correct=20))[0] == "success"
    assert ef.comparison_verdict(make_summary(n=60, agentic_correct=20, baseline_correct=40))[0] == "warning"
    mixed = make_summary(n=60)
    mixed["comparison"]["agentic_better"], mixed["comparison"]["baseline_better"] = ["a"], ["b"]
    assert ef.comparison_verdict(mixed)[0] == "info"
    one = make_summary()
    one["comparison"] = None
    assert "only one system" in ef.comparison_verdict(one)[1]


# ------------------------------------------------------------------------------ tables
def test_overview_facts():
    o = ef.overview(make_summary())
    assert o["Model"].startswith("openai/gpt-oss-20b") and o["Questions"] == "24 (3 per category)" and o["Question runs ok"] == "48/48"
    assert o["Duration"] == "15 min 39 s" and o["LLM judge"] == "on"


def test_retrieval_tables():
    rows = ef.retrieval_rows(make_summary())
    assert len(rows) == 4 and rows[0] == {"level": "Documents", "K": 3, "Recall@K": "1.000", "Precision@K": "0.958", "nDCG@K": "0.980"}
    assert {r["level"] for r in rows} == {"Documents", "Chunks"} and ef.retrieval_mrr(make_summary()) == {"Document MRR": "1.000", "Chunk MRR": "0.917"}


def test_answer_table_puts_both_systems_side_by_side_and_marks_missing_values():
    rows = {r["metric"]: r for r in ef.answer_rows(make_summary())}
    assert rows["Overall correct"]["Baseline RAG"] == "67% (16/24)" and rows["Overall correct"]["Agentic workflow"] == "58% (14/24)"
    assert rows["External information labelled"]["Baseline RAG"] == "n/a"        # den 0
    assert rows["Answer-point coverage (mean)"]["Agentic workflow"] == "50%" and rows["Restricted-content leaks"]["Baseline RAG"] == "0"
    one = make_summary()
    del one["modes"]["agentic"]
    assert {r["metric"]: r for r in ef.answer_rows(one)}["Overall correct"]["Agentic workflow"] == "n/a"


def test_routing_tables_come_from_the_agentic_run():
    rows = {r["metric"]: r["result"] for r in ef.routing_rows(make_summary())}
    assert rows["Private-first routing"] == "90% (19/21)" and rows["Unnecessary external search"] == "5% (1/21)" and rows["External fallback accuracy"] == "0% (0/3)"
    stats = {r["statistic"]: r["value"] for r in ef.external_stats(make_summary())}
    assert stats["Tavily calls in total"] == "6" and stats["Questions that needed a query rewrite"] == "9"
    assert ef.calibration_rows(make_summary())[0] == {"confidence range": "0.0-0.5", "questions": 8, "grade accuracy": "100%"}
    baseline_only = make_summary()
    del baseline_only["modes"]["agentic"]
    assert all(r["result"] == "n/a" for r in ef.routing_rows(baseline_only)) and ef.calibration_rows(baseline_only) == []


def test_system_and_comparison_tables_format_each_metric_by_kind():
    sysrows = {r["metric"]: r for r in ef.system_rows(make_summary(latency=2000))}
    assert sysrows["Average latency"]["Baseline RAG"] == "2.0 s" and sysrows["Total tokens"]["Agentic workflow"] == "104,668"
    assert sysrows["Estimated cost"]["Agentic workflow"] == "$0.0712" and sysrows["Error rate"]["Baseline RAG"] == "0% (0/24)"
    cmp = {r["metric"]: r for r in ef.comparison_rows(make_summary(latency=2000))}
    assert cmp["Overall correct rate"]["Baseline RAG"] == "67%" and cmp["Average latency (ms)"]["Agentic workflow"] == "18.0 s"
    assert cmp["Total tokens"]["Baseline RAG"] == "21,665" and cmp["Estimated cost (USD)"]["verdict"] == "baseline better"
    assert ef.comparison_findings(make_summary()) == ["Baseline is better on 1 metric(s): Overall correct rate."]


def test_category_table_and_chart():
    s = make_summary()
    assert ef.category_rows(s)[0] == {"category": "direct_kb", "Baseline RAG": "67% (2/3)", "Agentic workflow": "100% (3/3)"}
    chart = ef.category_chart(s)
    assert chart["category"] == ["direct_kb", "conflicting"] and chart["Baseline RAG"] == [66.7, 100.0] and chart["Agentic workflow"] == [100.0, 33.3]
    del s["modes"]["agentic"]["by_category"]["conflicting"]
    assert ef.category_chart(s)["Agentic workflow"] == [100.0, None]                   # missing is None, not 0


def test_failed_cases_table_and_titles():
    s = make_summary()
    row = ef.failed_rows(s)[0]
    assert row["case"] == "C01" and row["reason"] == "validation_failed" and row["answer points"] == "0/2"
    assert ef.failed_case_title(s["failed_cases"][0]) == "C01 · agentic · false_abstention"
    assert ef.failed_rows({}) == []
