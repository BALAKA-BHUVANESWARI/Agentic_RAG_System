"""Pure helpers that turn evaluation API payloads into display data (no Streamlit import, unit-testable).

Everything shown on the evaluation dashboard is computed here from the run's own ``summary.json``
(or the run-status payload). Nothing is hard-coded, and the wording never claims more than the
numbers support: small samples, partial runs, errors and model differences are called out.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from copilot.evaluation import variants as variant_registry
from copilot.evaluation.aggregate import ABLATION_METRICS
from copilot.evaluation.attacks import FAMILIES
from copilot.evaluation.paired import PAIRED_METRICS
from copilot.evaluation.schemas import CATEGORIES
from copilot.ui.formatting import format_ms

MODES = ("baseline", "agentic")
MODE_LABEL = {"baseline": "Baseline RAG", "agentic": "Agentic workflow"}
ACTIVE = frozenset({"running", "cancelling"})
RESUMABLE = frozenset({"cancelled", "stopped", "failed", "interrupted"})
SMALL_SAMPLE = 30  # fewer labelled questions than this per system: treat differences as noise

NA = "n/a"


# ------------------------------------------------------------------------------ which systems a run has
def systems_in(summary: dict) -> list[str]:
    """Systems that have results, in display order: ladder order (baseline first, full agent last), then any others."""
    present = [m for m, block in (summary.get("modes") or {}).items() if block]
    ladder = [n for n in variant_registry.NAMES if n in present]
    return ladder + [m for m in present if m not in ladder]


def table_systems(summary: dict) -> list[str]:
    """Columns for the side-by-side tables: every rung of an ablation run, otherwise always the two classic systems (missing values show n/a)."""
    return systems_in(summary) if summary.get("variants") else list(MODES)


def system_label(summary: dict, mode: str) -> str:
    """Display name. In an ablation run a rung is named by what it adds to the one before it."""
    if summary.get("variants"):
        try:
            return variant_registry.get_variant(mode).label
        except ValueError:
            return mode
    return MODE_LABEL.get(mode, mode)


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


def ablation_run_note(n_questions: int | None) -> str:
    """Caption for the 'ablation ladder' choice: it runs every rung on every selected question."""
    rungs = len(variant_registry.NAMES)
    scope = f"{n_questions} question(s) x {rungs} systems = up to {n_questions * rungs} question runs" if n_questions else f"every selected question on all {rungs} systems"
    return f"An ablation runs the whole ladder: {scope}. Real calls to Groq, Pinecone and Tavily; start with a small selection."


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
    if opts.get("attacks_only"):
        scope.append("adversarial questions only")
    if (opts.get("split") or "all") != "all":
        scope.append(f"{opts['split']} split")
    systems = " → ".join(systems_in(summary)) if summary.get("variants") else str(opts.get("mode", "both"))
    return {
        "Model": f"{s.get('llm_model', '?')} (reasoning: {s.get('reasoning_effort') or 'default'})",
        "Questions": f"{n_cases(summary)} ({'; '.join(scope) or 'whole dataset'})",
        "Systems": systems or str(opts.get("mode", "both")),
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
    present = systems_in(summary)
    if len(present) == 1:
        out.append(("info", f"Only the {system_label(summary, present[0])} has results, so there is no comparison."))
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
        if summary.get("variants"):
            return "info", "This ablation run does not include both the baseline and the full agentic workflow, so there is no baseline-vs-agentic comparison. See the Ablation tab."
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
        rows.append({"metric": label, **{system_label(summary, m): _cell(kind, (_mode(summary, m).get("answer") or {}).get(key)) for m in table_systems(summary)}})
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
        for m in table_systems(summary):
            value = (_mode(summary, m).get("system") or {}).get(key)
            row[system_label(summary, m)] = NA if value is None else fmt(value)
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
    for m in table_systems(summary):
        for c in (_mode(summary, m).get("by_category") or {}):
            if c not in cats:
                cats.append(c)
    return [
        {"category": c, **{system_label(summary, m): rate_text((_mode(summary, m).get("by_category") or {}).get(c)) for m in table_systems(summary)}}
        for c in cats
    ]


def category_chart(summary: dict) -> dict[str, list]:
    """Chart data (percent correct per category) as lists; None marks a missing value."""
    rows = category_rows(summary)
    data: dict[str, list] = {"category": [r["category"] for r in rows]}
    for m in table_systems(summary):
        by_cat = _mode(summary, m).get("by_category") or {}
        data[system_label(summary, m)] = [None if not by_cat.get(r["category"], {}).get("den") else round(by_cat[r["category"]]["value"] * 100, 1) for r in rows]
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


# ------------------------------------------------------------------------------ ablation ladder
def has_ablation(summary: dict) -> bool:
    """True when the run compared ablation rungs (it has an ``ablation`` block)."""
    return bool(summary.get("ablation"))


def _ladder_value(key: str, value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, dict):
        return rate_text(value)
    if key.endswith("_ms"):
        return format_ms(value)
    return count_text(value)


def ladder_rows(summary: dict) -> list[dict]:
    """The ablation ladder: one row per metric, one column per rung (each rung is named by what it adds)."""
    rows = (summary.get("ablation") or {}).get("rows") or []
    if not rows:
        return []
    return [
        {"metric": label, **{r["label"]: _ladder_value(key, (r.get("metrics") or {}).get(key)) for r in rows}}
        for key, label, _, _ in ABLATION_METRICS
    ]


def ladder_legend(summary: dict) -> list[dict]:
    """What each rung switches on, so the table headers can be read without the docs."""
    out = []
    for name in systems_in(summary):
        try:
            v = variant_registry.get_variant(name)
        except ValueError:
            continue
        out.append({"system": name, "label": v.label, "what it runs": v.description})
    return out


def ladder_findings(summary: dict) -> list[str]:
    """The plain-language findings produced with the ablation table (built from the numbers by the evaluator)."""
    return list((summary.get("ablation") or {}).get("findings") or [])


# ------------------------------------------------------------------------------ confidence intervals
def has_intervals(summary: dict) -> bool:
    """True when the run carries an ``intervals`` block."""
    return bool(summary.get("intervals"))


def interval_range(entry: dict | None) -> str:
    """'15–54%' for an entry with an interval, else an empty string."""
    if not entry or entry.get("lo") is None or entry.get("hi") is None:
        return ""
    return f"{entry['lo'] * 100:.0f}–{entry['hi'] * 100:.0f}%"


def interval_text(entry: dict | None) -> str:
    """``{value, num, den, lo, hi}`` -> '67% (16/24) [47–83%]'; the interval is left out when it was not computed."""
    if not entry or not entry.get("den"):
        return NA
    text = f"{pct(entry.get('value'))} ({entry['num']}/{entry['den']})"
    rng = interval_range(entry)
    return f"{text} [{rng}]" if rng else text


def interval_rows(summary: dict) -> list[dict]:
    """Each headline rate with its Wilson interval, one column per system."""
    rates = (summary.get("intervals") or {}).get("rates") or {}
    systems = [m for m in systems_in(summary) if rates.get(m)]
    keys = [(k, label) for k, label, _, _ in PAIRED_METRICS if any(k in rates[m] for m in systems)]
    return [{"metric": label, **{system_label(summary, m): interval_text(rates[m].get(k)) for m in systems}} for k, label in keys]


def _signed_points(value: float | None) -> str:
    return NA if value is None else f"{value * 100:+.0f}"


_HIGHER_IS_BETTER = {key: higher for key, _, higher, _ in PAIRED_METRICS}


def paired_rows(summary: dict) -> list[dict]:
    """Per-question paired differences between systems that answered the same questions.

    ``better on`` / ``worse on`` count questions where the later system did better / worse than the earlier one,
    already corrected for metrics where lower is better (hallucinations, false abstentions).
    """
    rows = []
    for pair in (summary.get("intervals") or {}).get("paired") or []:
        for m in pair.get("metrics") or []:
            higher = _HIGHER_IS_BETTER.get(m["key"], True)
            better_on, worse_on = (m["gained"], m["lost"]) if higher else (m["lost"], m["gained"])
            rows.append({
                "comparison": f"{pair['from']} → {pair['to']}", "metric": m["label"], "questions": m["n"],
                "before": pct(m.get("before")), "after": pct(m.get("after")),
                "difference (points)": _signed_points(m.get("diff")),
                "interval (points)": f"[{_signed_points(m.get('lo'))}, {_signed_points(m.get('hi'))}]" if m.get("lo") is not None else NA,
                "better on": better_on, "worse on": worse_on, "verdict": m.get("verdict", ""),
            })
    return rows


def intervals_note(summary: dict) -> str:
    """How the intervals were computed and what they do not cover."""
    iv = summary.get("intervals") or {}
    if not iv:
        return ""
    return (f"{pct(iv.get('confidence'))} intervals. Rates: Wilson score. Differences: paired bootstrap over questions "
            f"({iv.get('resamples', 0):,} resamples, seed {iv.get('seed')}). {iv.get('caveat') or ''}").strip()


def intervals_verdict(summary: dict) -> tuple[str, str]:
    """(level, text) on what the paired intervals support. Never names a winner; says so when the sample is too small."""
    iv = summary.get("intervals")
    if not iv:
        return "info", "No interval data in this run."
    metrics = [m for p in iv.get("paired") or [] for m in p.get("metrics") or []]
    if not metrics:
        return "info", "This run has a single system, so there is no paired comparison. The per-system intervals are shown below."
    verdicts = [m.get("verdict") for m in metrics]
    decided = [v for v in verdicts if v in ("better", "worse")]
    if all(v == "too few questions" for v in verdicts):
        return "warning", "Too few questions for an interval to mean anything. Run more questions before reading any difference."
    if not decided:
        return "info", f"No paired difference is distinguishable from zero at {pct(iv.get('confidence'))}: with this many questions the data cannot separate the systems."
    return "info", (f"{len(decided)} of {len(metrics)} paired differences have an interval that excludes zero "
                    f"({verdicts.count('better')} better, {verdicts.count('worse')} worse). That says the questions in this dataset favour one system, not that repeat runs would agree.")


# ------------------------------------------------------------------------------ attack success
_FLAG_COLUMN = {
    "leak": "leaked restricted content", "answered_when_should_abstain": "answered instead of refusing", "unwanted_web_search": "unwanted web search",
}


def has_attacks(summary: dict) -> bool:
    """True when the run contained adversarial questions."""
    return bool((summary.get("attacks") or {}).get("modes"))


def _attack_modes(summary: dict) -> list[str]:
    modes = (summary.get("attacks") or {}).get("modes") or {}
    return [m for m in systems_in(summary) if m in modes]


def attack_rows(summary: dict) -> list[dict]:
    """One row per system: how many attacks were judged, how many succeeded (with an interval) and which detector fired."""
    at = summary.get("attacks") or {}
    flags = list((at.get("flags") or _FLAG_COLUMN))
    rows = []
    for m in _attack_modes(summary):
        b = at["modes"][m]
        rows.append({
            "system": system_label(summary, m), "attacks": b["attacks"], "judged": b["judged"], "not judged (errors)": b["not_judged_errors"],
            "succeeded": b["succeeded"], "attack success": interval_text(b.get("asr")),
            **{_FLAG_COLUMN.get(f, f): (b.get("by_flag") or {}).get(f, 0) for f in flags},
            "held, answer incorrect": b.get("held_but_incorrect", 0),
        })
    return rows


def attack_family_rows(summary: dict) -> list[dict]:
    """Succeeded/judged per attack family and system."""
    modes = (summary.get("attacks") or {}).get("modes") or {}
    systems = _attack_modes(summary)
    order = [k for k, _, _, _ in FAMILIES] + ["other"]
    present = [k for k in order if any(k in (modes[m].get("by_family") or {}) for m in systems)]
    rows = []
    for key in present:
        label = next((modes[m]["by_family"][key]["label"] for m in systems if key in (modes[m].get("by_family") or {})), key)
        row = {"attack family": label}
        for m in systems:
            fam = (modes[m].get("by_family") or {}).get(key)
            row[system_label(summary, m)] = f"{fam['succeeded']}/{fam['judged']}" if fam else NA
        rows.append(row)
    return rows


def attack_succeeded_rows(summary: dict) -> list[dict]:
    """Every attack that tripped a detector, with the system, family and what was detected."""
    at = summary.get("attacks") or {}
    descriptions = at.get("flags") or {}
    labels = {k: label for k, label, _, _ in FAMILIES} | {"other": "Other adversarial cases"}
    return [
        {"system": system_label(summary, m), "case": c["case_id"], "family": labels.get(c.get("family"), c.get("family")),
         "detected": "; ".join(descriptions.get(f, f) for f in c.get("flags") or []), "question": c.get("question", "")}
        for m in _attack_modes(summary) for c in at["modes"][m].get("succeeded_cases") or []
    ]


def attack_banners(summary: dict) -> list[tuple[str, str]]:
    """(level, text) lines: one per system, then the lower-bound caveat. Wording follows the counts; it never says a system is 'secure'."""
    at = summary.get("attacks") or {}
    out: list[tuple[str, str]] = []
    for m in _attack_modes(summary):
        b = at["modes"][m]
        name = system_label(summary, m)
        if not b["judged"]:
            out.append(("warning", f"{name}: none of the {b['attacks']} attacks could be judged (service errors)."))
            continue
        rng = interval_range(b.get("asr"))
        rate_part = pct(b["asr"].get("value")) + (f", {pct(at.get('confidence'))} interval {rng}" if rng else "")
        text = f"{name}: {b['succeeded']} of {b['judged']} judged attacks tripped a detector ({rate_part})."
        if b["not_judged_errors"]:
            text += f" {b['not_judged_errors']} more could not be judged because the service failed."
        out.append(("warning" if b["succeeded"] else "info", text))
    if at.get("caveat"):
        out.append(("info", at["caveat"]))
    return out
