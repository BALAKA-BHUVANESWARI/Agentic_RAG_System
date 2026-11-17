"""Aggregate per-case results into run-level metrics, and compare baseline with agentic.

Every rate is reported as ``{"value", "num", "den"}`` so the dashboard can show the sample size next
to the percentage. With a small dataset these numbers are noisy: read ``den`` before concluding.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

from copilot.config import Settings
from copilot.evaluation.results import CaseResult, RetrievalCaseResult
from copilot.evaluation.schemas import CATEGORIES, EvalCase


def rate(num: int, den: int) -> dict[str, Any]:
    """A rate with its counts (value is None when the denominator is zero)."""
    return {"value": (num / den) if den else None, "num": num, "den": den}


def mean(values: Iterable[float | None]) -> float | None:
    """Mean ignoring None."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0-100)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def estimate_cost(prompt_tokens: int, completion_tokens: int, tavily_calls: int, settings: Settings) -> float:
    """Approximate list-price cost in USD (prices are configurable; a free plan actually costs nothing)."""
    return (prompt_tokens * settings.price_llm_input_per_mtok + completion_tokens * settings.price_llm_output_per_mtok) / 1e6 + tavily_calls * settings.price_tavily_per_search


def _rate_of(items: list[tuple[EvalCase, CaseResult]], key: str, where: Callable[[EvalCase, CaseResult], bool] = lambda c, r: True) -> dict[str, Any]:
    vals = [r.scores.get(key) for c, r in items if where(c, r) and r.scores.get(key) is not None]
    return rate(sum(1 for v in vals if v), len(vals))


def summarize_retrieval(results: list[RetrievalCaseResult], ks: list[int]) -> dict[str, Any]:
    """Mean of each retrieval metric over the cases that have expected documents."""
    scored = [r for r in results if r.scores]
    keys = sorted({k for r in scored for k in r.scores})
    return {"n_cases": len(scored), "metrics": {k: mean(r.scores.get(k) for r in scored) for k in keys}, "ks": ks}


def summarize_mode(cases: list[EvalCase], results: list[CaseResult], settings: Settings) -> dict[str, Any]:
    """Answer, routing and system metrics for one system (baseline or agentic)."""
    by_id = {c.id: c for c in cases}
    items = [(by_id[r.case_id], r) for r in results if r.case_id in by_id]
    answerable = [(c, r) for c, r in items if not c.expected_abstention]
    abstain = [(c, r) for c, r in items if c.expected_abstention]
    scored_pts = [r.scores["point_coverage"] for c, r in answerable if r.scores.get("point_coverage") is not None]

    answer = {
        "overall_correct_rate": rate(sum(1 for c, r in items if r.scores.get("correct")), len(items)),
        "answer_correct_rate": rate(sum(1 for c, r in answerable if r.scores.get("correct")), len(answerable)),
        "answer_point_coverage_mean": mean(scored_pts),
        "false_abstention_rate": rate(sum(1 for c, r in answerable if r.abstained), len(answerable)),
        "correct_abstention_rate": rate(sum(1 for c, r in abstain if r.scores.get("correct")), len(abstain)),
        "hallucinated_answer_rate": rate(sum(1 for c, r in abstain if r.scores.get("outcome") == "hallucinated_answer"), len(abstain)),
        "citation_correct_rate": _rate_of(items, "citation_correct"),
        "conflict_handling_rate": rate(sum(1 for c, r in items if c.category == "conflicting" and r.scores.get("correct")), sum(1 for c, r in items if c.category == "conflicting")),
        "external_label_rate": _rate_of(items, "external_label_ok"),
        "restricted_content_leaks": sum(1 for c, r in items if r.scores.get("leaked_keywords")),
        "faithfulness_rate": _rate_of(items, "faithful"),
        "unsupported_claim_rate": _rate_of(items, "unsupported_claims"),
        "answer_relevance_mean": mean(r.scores.get("answer_relevance") for c, r in items),
    }

    agentic = [(c, r) for c, r in items if r.mode == "agentic"]
    routing: dict[str, Any] = {}
    if agentic:
        not_ext = [(c, r) for c, r in agentic if not c.expects_external]
        private_only = [(c, r) for c, r in agentic if c.expected_route == "private_only"]
        graded = [(c, r) for c, r in agentic if r.scores.get("grading_correct") is not None and r.grade_confidence is not None]
        bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.01)]
        calib = []
        for lo, hi in bins:
            sel = [(c, r) for c, r in graded if lo <= r.grade_confidence < hi]
            calib.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(sel), "accuracy": mean(1.0 if r.scores["grading_correct"] else 0.0 for c, r in sel)})
        routing = {
            "private_first_routing_accuracy": _rate_of(agentic, "private_first_ok"),
            "correct_private_only_answer_rate": rate(sum(1 for c, r in private_only if r.scores.get("correct") and not r.external_triggered), len(private_only)),
            "unnecessary_external_search_rate": rate(sum(1 for c, r in not_ext if r.scores.get("unnecessary_external")), len(not_ext)),
            "external_fallback_accuracy": _rate_of(agentic, "external_fallback_ok"),
            "query_rewrite_success_rate": _rate_of(agentic, "rewrite_success"),
            "evidence_grading_accuracy": _rate_of(agentic, "grading_correct"),
            "grader_confidence_calibration": calib,
            "mean_confidence_when_correct": mean(r.grade_confidence for c, r in graded if r.scores["grading_correct"]),
            "mean_confidence_when_wrong": mean(r.grade_confidence for c, r in graded if not r.scores["grading_correct"]),
            "correct_abstention_rate": answer["correct_abstention_rate"],
            "final_validation_failure_rate": rate(sum(1 for c, r in agentic if r.scores.get("validation_failed")), len(agentic)),
            "external_searches_total": sum(1 for c, r in agentic if r.external_triggered),
            "tavily_calls_total": sum(r.tavily_calls for c, r in agentic),
            "cases_with_rewrite": sum(1 for c, r in agentic if r.retry_count > 0),
        }

    lat = [r.latency_ms for _, r in items]
    pt, ct, tv = (sum(getattr(r, f) for _, r in items) for f in ("prompt_tokens", "completion_tokens", "tavily_calls"))
    system = {
        "n_cases": len(items), "latency_avg_ms": mean(lat), "latency_p95_ms": percentile(lat, 95),
        "error_rate": rate(sum(1 for _, r in items if r.error or r.abstention_reason == "internal_error"), len(items)),
        "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "llm_calls": sum(r.llm_calls for _, r in items),
        "tavily_calls": tv, "estimated_cost_usd": estimate_cost(pt, ct, tv, settings),
    }
    by_category = {}
    for cat in CATEGORIES:
        sel = [(c, r) for c, r in items if c.category == cat]
        if sel:
            by_category[cat] = rate(sum(1 for c, r in sel if r.scores.get("correct")), len(sel))
    return {"answer": answer, "routing": routing, "system": system, "by_category": by_category}


# Metrics compared between baseline and agentic: (label, path in summary, higher_is_better)
COMPARED: list[tuple[str, tuple[str, str], bool]] = [
    ("Overall correct rate", ("answer", "overall_correct_rate"), True),
    ("Answer correct rate (answerable questions)", ("answer", "answer_correct_rate"), True),
    ("Answer point coverage (mean)", ("answer", "answer_point_coverage_mean"), True),
    ("Correct abstention rate (unanswerable / out of scope)", ("answer", "correct_abstention_rate"), True),
    ("Hallucinated answer rate (should abstain but answered)", ("answer", "hallucinated_answer_rate"), False),
    ("False abstention rate (answerable but abstained)", ("answer", "false_abstention_rate"), False),
    ("Citation correct rate", ("answer", "citation_correct_rate"), True),
    ("Conflict handling rate", ("answer", "conflict_handling_rate"), True),
    ("Unsupported claim rate (judge)", ("answer", "unsupported_claim_rate"), False),
    ("Average latency (ms)", ("system", "latency_avg_ms"), False),
    ("Total tokens", ("system", "total_tokens"), False),
    ("Estimated cost (USD)", ("system", "estimated_cost_usd"), False),
]


def _value(metric: Any) -> float | None:
    return metric.get("value") if isinstance(metric, dict) else metric


def compare_modes(baseline: dict[str, Any], agentic: dict[str, Any], tolerance: float = 0.02) -> dict[str, Any]:
    """Side-by-side rows plus plain-language findings built only from the numbers.

    Differences smaller than ``tolerance`` (absolute, on 0-1 rates) are called ties. No significance
    test is run: with ~50 questions a gap of a few points can easily be noise.
    """
    rows, better, worse, tied = [], [], [], []
    for label, (section, key), higher in COMPARED:
        b, a = _value(baseline[section].get(key)), _value(agentic[section].get(key))
        if b is None or a is None:
            rows.append({"metric": label, "baseline": b, "agentic": a, "verdict": "n/a"})
            continue
        scale = 1.0 if key.endswith("_rate") or key.endswith("coverage_mean") else max(abs(b), abs(a), 1e-9)
        delta = (a - b) / scale
        verdict = "tie" if abs(delta) < (tolerance if scale == 1.0 else 0.05) else ("agentic better" if (delta > 0) == higher else "baseline better")
        rows.append({"metric": label, "baseline": b, "agentic": a, "verdict": verdict})
        {"agentic better": better, "baseline better": worse, "tie": tied}[verdict].append(label)
    notes = [
        f"Agentic is better on {len(better)} metric(s): {', '.join(better) or 'none'}.",
        f"Baseline is better on {len(worse)} metric(s): {', '.join(worse) or 'none'}.",
        f"{len(tied)} metric(s) are effectively tied.",
        "These are single-run results on a small labelled set; treat differences of a few points as noise until repeated.",
    ]
    return {"rows": rows, "findings": notes, "agentic_better": better, "baseline_better": worse, "tied": tied}
