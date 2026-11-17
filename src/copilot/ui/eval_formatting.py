"""Pure helpers that turn evaluation API payloads into display data (no Streamlit import, unit-testable).

Everything shown on the evaluation dashboard is computed here from the run's own ``summary.json``
(or the run-status payload). Nothing is hard-coded, and the wording never claims more than the
numbers support: small samples, partial runs, errors and model differences are called out.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from copilot.evaluation.schemas import CATEGORIES
from copilot.ui.formatting import format_ms

MODES = ("baseline", "agentic")
MODE_LABEL = {"baseline": "Baseline RAG", "agentic": "Agentic workflow"}
ACTIVE = frozenset({"running", "cancelling"})
RESUMABLE = frozenset({"cancelled", "stopped", "failed", "interrupted"})
SMALL_SAMPLE = 30  # fewer labelled questions than this per system: treat differences as noise

NA = "n/a"


# ------------------------------------------------------------------------------ number formatting
def pct(value: float | None) -> str:
    """0.667 -> '67%'."""
    return NA if value is None else f"{value * 100:.0f}%"


def rate_text(rate: dict | None) -> str:
    """A ``{value, num, den}`` rate -> '67% (16/24)'. No denominator means not applicable."""
    if not rate or not rate.get("den"):
        return NA
    return f"{pct(rate.get('value'))} ({rate['num']}/{rate['den']})"


def number_text(value: float | int | None, digits: int = 2) -> str:
    """Plain number, or 'n/a'."""
    return NA if value is None else f"{value:,.{digits}f}"


def count_text(value: int | None) -> str:
    """Integer with thousands separators, or 'n/a'."""
    return NA if value is None else f"{int(value):,}"


def cost_text(value: float | None) -> str:
    """Approximate USD cost."""
    return NA if value is None else f"${value:.4f}"


def metric_value_text(metric: str, value: float | None) -> str:
    """Format a comparison-table value by what the metric measures (taken from its label)."""
    if value is None:
        return NA
    name = metric.lower()
    if "(ms)" in name:
        return format_ms(value)
    if "token" in name:
        return count_text(value)
    if "cost" in name:
        return cost_text(value)
    return pct(value)


def _parse_time(text: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def duration_text(started: str | None, finished: str | None) -> str:
    """'15 min 39 s' between two ISO timestamps; '-' when unknown."""
    a, b = _parse_time(started), _parse_time(finished)
    if not a or not b:
        return "-"
    seconds = max(0, int((b - a).total_seconds()))
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


# ------------------------------------------------------------------------------ run list and status
def run_label(run: dict) -> str:
    """One-line label for the run picker."""
    return f"{run['run_id']} · {run['status']} · {run.get('mode') or '?'} · {run.get('num_cases') or '?'} cases"


def run_rows(runs: list[dict]) -> list[dict]:
    """Rows for the run-history table."""
    return [
        {
            "run": r["run_id"], "status": r["status"], "systems": r.get("mode") or "-", "cases": r.get("num_cases"),
            "progress": f"{r.get('done', 0)}/{r.get('total', 0)}", "judge": "yes" if r.get("use_judge") else "no",
            "started": (r.get("started_at") or "")[:19].replace("T", " "),
        }
        for r in runs
    ]


def progress_fraction(info: dict) -> float:
    """0..1 for a progress bar."""
    total = info.get("total") or 0
    return 0.0 if total <= 0 else max(0.0, min(1.0, (info.get("done") or 0) / total))


def status_banner(info: dict) -> tuple[str, str]:
    """(level, text) for a run's status; level is one of info / success / warning / error."""
    status, done, total = info["status"], info.get("done", 0), info.get("total", 0)
    if status == "running":
        return "info", f"Running: {done}/{total} question runs finished. {info.get('message') or ''}".strip()
    if status == "cancelling":
        return "warning", "Cancelling: the question in progress will finish first."
    if status == "completed":
        return "success", f"Completed: {done}/{total} question runs finished."
    if status == "cancelled":
        return "warning", f"Cancelled at {done}/{total}. The numbers below cover only what ran. Resume the run to finish it."
    if status == "stopped":
        return "error", f"Stopped at {done}/{total} after repeated service errors (usually a provider quota or outage). Resume it once the service is available."
    if status == "failed":
        return "error", f"Failed: {info.get('error') or 'unknown error'}. Partial results are kept; resume to retry."
    if status == "interrupted":
        return "warning", "The server stopped while this run was in progress. Resume it to continue."
    return "info", f"Status: {status}"


def resumable_runs(runs: list[dict]) -> list[dict]:
    """Runs that can be resumed."""
    return [r for r in runs if r["status"] in RESUMABLE]


def active_run(runs: list[dict]) -> dict | None:
    """The run in progress, if any."""
    return next((r for r in runs if r["status"] in ACTIVE), None)


def latest_completed(runs: list[dict]) -> dict | None:
    """Newest completed run (the API lists newest first)."""
    return next((r for r in runs if r["status"] == "completed"), None)


# ------------------------------------------------------------------------------ pre-run estimate
def selected_question_count(per_category: int | None, limit: int | None, categories: list[str] | None) -> int | None:
    """Upper bound on the number of questions a run will use; None when the whole dataset is selected."""
    if not per_category and not limit:
        return None
    n = (per_category * len(categories or CATEGORIES)) if per_category else None
    if limit:
        n = min(n, limit) if n else limit
    return n


def estimate_tokens(summary: dict | None, n_questions: int | None, mode: str, use_judge: bool) -> int | None:
    """Rough token need of a run, scaled from an earlier run's measured usage. None when it cannot be estimated."""
    if not summary or not n_questions:
        return None
    total = 0
    for m in (MODES if mode == "both" else (mode,)):
        system = (summary.get("modes", {}).get(m) or {}).get("system") or {}
        if not system.get("n_cases") or system.get("total_tokens") is None:
            return None
        total += system["total_tokens"] / system["n_cases"] * n_questions
    judge = summary.get("judge") or {}
    if use_judge and judge.get("judged_answers"):
        total += judge["tokens"] / judge["judged_answers"] * n_questions
    return int(total)


# ------------------------------------------------------------------------------ overview and caveats
def n_cases(summary: dict) -> int:
    """Number of labelled questions in the run."""
    return int((summary.get("dataset") or {}).get("num_cases") or 0)


def overview(summary: dict) -> dict[str, str]:
    """Header facts about the run."""
    s, opts, prog = summary.get("settings") or {}, summary.get("options") or {}, summary.get("progress") or {}
    scope = []
    if opts.get("per_category"):
        scope.append(f"{opts['per_category']} per category")
    if opts.get("limit"):
        scope.append(f"first {opts['limit']}")
    if opts.get("categories"):
        scope.append(", ".join(opts["categories"]))
    return {
        "Model": f"{s.get('llm_model', '?')} (reasoning: {s.get('reasoning_effort') or 'default'})",
        "Questions": f"{n_cases(summary)} ({'; '.join(scope) or 'whole dataset'})",
        "Systems": str(opts.get("mode", "both")),
        "Question runs ok": f"{prog.get('units_ok', 0)}/{prog.get('units_total', 0)}",
        "Duration": duration_text(summary.get("started_at"), summary.get("finished_at")),
        "LLM judge": "on" if (summary.get("judge") or {}).get("enabled") else "off",
    }


def caveats(summary: dict) -> list[tuple[str, str]]:
    """Plain-language warnings about how far this run's numbers can be trusted (level, text)."""
    out: list[tuple[str, str]] = []
    status, prog = summary.get("status"), summary.get("progress") or {}
    if status and status != "completed":
        out.append(("warning", f"This run is {status}: the metrics cover only the question runs that finished."))
    missing = (prog.get("units_total") or 0) - (prog.get("units_ok") or 0)
    if missing > 0:
        out.append(("warning", f"{missing} question run(s) did not finish without an error. Failed requests count as incorrect, so they lower the scores."))
    present = [m for m in MODES if (summary.get("modes") or {}).get(m)]
    if len(present) == 1:
        out.append(("info", f"Only the {MODE_LABEL[present[0]]} has results, so there is no comparison."))
    n = n_cases(summary)
    if 0 < n < SMALL_SAMPLE:
        out.append(("info", f"Only {n} labelled questions per system: differences of a few points are noise. Repeat runs before drawing conclusions."))
    model = (summary.get("settings") or {}).get("llm_model")
    if model:
        out.append(("info", f"Generated with {model}. Runs on different models or settings are not comparable."))
    if (summary.get("judge") or {}).get("enabled"):
        out.append(("info", "Faithfulness and relevance come from an LLM judge in the same model family as the generator: treat them as a noisy signal."))
    slow = [
        (summary["modes"][m].get("system") or {}).get("latency_avg_ms") or 0 for m in present
    ]
    if slow and max(slow) > 10_000:
        out.append(("info", "Latency includes waiting on provider rate limits (common on free tiers); token counts are the steadier cost signal."))
    return out


def comparison_verdict(summary: dict) -> tuple[str, str]:
    """(level, text) on whether the comparison supports a conclusion. Never says the agentic system is better unless the data do and the sample is large enough."""
    comp = summary.get("comparison")
    if not comp:
        return "info", "No comparison: this run has results for only one system."
    n = n_cases(summary)
    better_a, better_b = comp.get("agentic_better") or [], comp.get("baseline_better") or []
    if n < SMALL_SAMPLE:
        return "warning", f"Not enough evidence to conclude which system is better: {n} questions per system. Agentic wins {len(better_a)} metric(s), baseline wins {len(better_b)}, the rest tie."
    if better_a and not better_b:
        return "success", f"On {n} questions the agentic workflow is better on {len(better_a)} metric(s) and worse on none. Check the sample sizes below."
    if better_b and not better_a:
        return "warning", f"On {n} questions the baseline is better on {len(better_b)} metric(s) and worse on none."
    return "info", f"Mixed result on {n} questions: agentic better on {len(better_a)} metric(s), baseline better on {len(better_b)}."


# ------------------------------------------------------------------------------ tables
def _mode(summary: dict, mode: str) -> dict:
    return (summary.get("modes") or {}).get(mode) or {}


def retrieval_rows(summary: dict) -> list[dict]:
    """Recall / precision / nDCG at each K, for documents and chunks."""
    ret = summary.get("retrieval") or {}
    metrics = ret.get("metrics") or {}
    rows = []
    for level, label in (("doc", "Documents"), ("chunk", "Chunks")):
        for k in ret.get("ks") or []:
            rows.append({
                "level": label, "K": k,
                "Recall@K": number_text(metrics.get(f"{level}_recall@{k}"), 3),
                "Precision@K": number_text(metrics.get(f"{level}_precision@{k}"), 3),
                "nDCG@K": number_text(metrics.get(f"{level}_ndcg@{k}"), 3),
            })
    return rows


def retrieval_mrr(summary: dict) -> dict[str, str]:
    """Mean reciprocal rank at document and chunk level."""
    m = (summary.get("retrieval") or {}).get("metrics") or {}
    return {"Document MRR": number_text(m.get("doc_mrr"), 3), "Chunk MRR": number_text(m.get("chunk_mrr"), 3)}


_ANSWER_METRICS: list[tuple[str, str, str]] = [
    ("overall_correct_rate", "Overall correct", "rate"),
    ("answer_correct_rate", "Correct (answerable questions)", "rate"),
    ("answer_point_coverage_mean", "Answer-point coverage (mean)", "mean"),
    ("false_abstention_rate", "False abstentions (answerable but abstained)", "rate"),
    ("correct_abstention_rate", "Correct abstentions", "rate"),
    ("hallucinated_answer_rate", "Hallucinated (should have abstained)", "rate"),
    ("citation_correct_rate", "Citation correct", "rate"),
    ("conflict_handling_rate", "Conflicting documents communicated", "rate"),
    ("external_label_rate", "External information labelled", "rate"),
    ("faithfulness_rate", "Faithful (LLM judge)", "rate"),
    ("unsupported_claim_rate", "Unsupported claims (LLM judge)", "rate"),
    ("answer_relevance_mean", "Relevance (LLM judge, mean)", "mean"),
    ("restricted_content_leaks", "Restricted-content leaks", "count"),
]


def _cell(kind: str, value: Any) -> str:
    if value is None:
        return NA
    return {"rate": rate_text, "mean": lambda v: pct(v), "count": lambda v: str(v)}[kind](value)


def answer_rows(summary: dict) -> list[dict]:
    """Answer-quality metrics, baseline next to agentic."""
    rows = []
    for key, label, kind in _ANSWER_METRICS:
        rows.append({"metric": label, **{MODE_LABEL[m]: _cell(kind, (_mode(summary, m).get("answer") or {}).get(key)) for m in MODES}})
    return rows


_ROUTING_METRICS: list[tuple[str, str]] = [
    ("private_first_routing_accuracy", "Private-first routing"),
    ("correct_private_only_answer_rate", "Correct private-only answers"),
    ("unnecessary_external_search_rate", "Unnecessary external search"),
    ("external_fallback_accuracy", "External fallback accuracy"),
    ("query_rewrite_success_rate", "Query rewrite success"),
    ("evidence_grading_accuracy", "Evidence grading accuracy"),
    ("correct_abstention_rate", "Correct abstention"),
    ("final_validation_failure_rate", "Final validation failures"),
]


def routing_rows(summary: dict) -> list[dict]:
    """Routing metrics (the agentic workflow only; the baseline has no routing)."""
    routing = _mode(summary, "agentic").get("routing") or {}
    return [{"metric": label, "result": rate_text(routing.get(key))} for key, label in _ROUTING_METRICS]


def external_stats(summary: dict) -> list[dict]:
    """Counts of external searches and rewrites in the agentic run."""
    r = _mode(summary, "agentic").get("routing") or {}
    return [
        {"statistic": "Questions that triggered an external search", "value": count_text(r.get("external_searches_total"))},
        {"statistic": "Tavily calls in total", "value": count_text(r.get("tavily_calls_total"))},
        {"statistic": "Questions that needed a query rewrite", "value": count_text(r.get("cases_with_rewrite"))},
    ]


def calibration_rows(summary: dict) -> list[dict]:
    """Grader confidence vs how often the grade was right. The score is a decision signal, not a calibrated probability."""
    routing = _mode(summary, "agentic").get("routing") or {}
    return [{"confidence range": b["range"], "questions": b["n"], "grade accuracy": pct(b.get("accuracy"))} for b in routing.get("grader_confidence_calibration") or []]


def system_rows(summary: dict) -> list[dict]:
    """Latency, errors, tokens and approximate cost per system."""
    specs: list[tuple[str, str, Any]] = [
        ("Questions answered", "n_cases", count_text),
        ("Error rate", "error_rate", rate_text),
        ("Average latency", "latency_avg_ms", format_ms),
        ("P95 latency", "latency_p95_ms", format_ms),
        ("Total tokens", "total_tokens", count_text),
        ("Prompt tokens", "prompt_tokens", count_text),
        ("Completion tokens", "completion_tokens", count_text),
        ("LLM calls", "llm_calls", count_text),
        ("Tavily calls", "tavily_calls", count_text),
        ("Estimated cost", "estimated_cost_usd", cost_text),
    ]
    rows = []
    for label, key, fmt in specs:
        row = {"metric": label}
        for m in MODES:
            value = (_mode(summary, m).get("system") or {}).get(key)
            row[MODE_LABEL[m]] = NA if value is None else fmt(value)
        rows.append(row)
    return rows


def comparison_rows(summary: dict) -> list[dict]:
    """The baseline-vs-agentic table from the run's own comparison."""
    comp = summary.get("comparison") or {}
    return [
        {"metric": r["metric"], MODE_LABEL["baseline"]: metric_value_text(r["metric"], r.get("baseline")),
         MODE_LABEL["agentic"]: metric_value_text(r["metric"], r.get("agentic")), "verdict": r.get("verdict", "")}
        for r in comp.get("rows") or []
    ]


def comparison_findings(summary: dict) -> list[str]:
    """The plain-language findings produced with the comparison."""
    return list((summary.get("comparison") or {}).get("findings") or [])


def category_rows(summary: dict) -> list[dict]:
    """Overall-correct rate per question category, both systems."""
    cats: list[str] = []
    for m in MODES:
        for c in (_mode(summary, m).get("by_category") or {}):
            if c not in cats:
                cats.append(c)
    return [
        {"category": c, **{MODE_LABEL[m]: rate_text((_mode(summary, m).get("by_category") or {}).get(c)) for m in MODES}}
        for c in cats
    ]


def category_chart(summary: dict) -> dict[str, list]:
    """Chart data (percent correct per category) as lists; None marks a missing value."""
    rows = category_rows(summary)
    data: dict[str, list] = {"category": [r["category"] for r in rows]}
    for m in MODES:
        by_cat = _mode(summary, m).get("by_category") or {}
        data[MODE_LABEL[m]] = [None if not by_cat.get(r["category"], {}).get("den") else round(by_cat[r["category"]]["value"] * 100, 1) for r in rows]
    return data


def failed_rows(summary: dict) -> list[dict]:
    """Table rows for the failed or incorrect cases."""
    return [
        {"case": c["case_id"], "system": c["mode"], "category": c["category"], "outcome": c["outcome"], "reason": c.get("abstention_reason") or c.get("error") or "",
         "answer points": f"{c.get('points_met', 0)}/{c.get('points_total', 0)}", "question": c["question"]}
        for c in summary.get("failed_cases") or []
    ]


def failed_case_title(case: dict) -> str:
    """Expander title for one failed case."""
    return f"{case['case_id']} · {case['mode']} · {case['outcome']}"
