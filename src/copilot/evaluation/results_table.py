"""Render the README results tables from saved runs (reads ``summary.json`` only; no LLM, search or network calls).

Why this exists: the README table used to be typed by hand from one run. A table generated from the run's own file
cannot drift from it. The renderer also refuses what would mislead: a run that did not finish (unless asked), runs
whose settings differ when pooling, and the same questions counted twice.

Pooling adds the counts behind each rate (num and den) across runs and recomputes the percentage; it never averages
percentages. P95 latency cannot be pooled and is shown as n/a. Intervals are Wilson intervals computed from the same
counts, so they exist for every run, including runs saved before the summary carried an ``intervals`` block.

Standard library plus ``copilot.evaluation.stats`` only, so it runs wherever Python runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from copilot.evaluation.stats import wilson_interval

SMALL_SAMPLE = 30  # keep equal to copilot.ui.eval_formatting.SMALL_SAMPLE (a unit test checks it)
CONFIDENCE = 0.95
NA = "n/a"
MODE_ORDER = ("baseline", "grade", "agentic")
MODE_LABEL = {"baseline": "Baseline RAG", "grade": "Grader only (ablation rung)", "agentic": "Agentic workflow"}

# The `conflicting` category (C01-C06 and F29-F31) expects the older edition of the password policy to be stated as a live conflict, but that
# document is now marked superseded, so a system that follows its status states only the current rule (docs/FINDINGS.md, finding 10).
# Pre-registered 2026-11-22, before any run on the fresh block or the test split: results are shown with and without the category, and no
# claim about conflicting documents is made. Excluded by the category recorded in cases.jsonl, not by a list of ids.
CONFLICTING_CATEGORY = "conflicting"
WITHOUT_CONFLICTING = ("answer", "overall_correct_without_conflicting")
DERIVED_KEYS = frozenset({WITHOUT_CONFLICTING[1]})

# (row label, summary section, metric key). A system without the metric (the baseline has no routing) shows n/a.
RATE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Overall correct", "answer", "overall_correct_rate"),
    ("Overall correct, without the conflicting category", "answer", WITHOUT_CONFLICTING[1]),
    ("Answerable questions answered correctly", "answer", "answer_correct_rate"),
    ("False abstentions", "answer", "false_abstention_rate"),
    ("Correct abstentions", "answer", "correct_abstention_rate"),
    ("Hallucinated answers (should have abstained)", "answer", "hallucinated_answer_rate"),
    ("Citation correct", "answer", "citation_correct_rate"),
    ("Faithful (LLM judge)", "answer", "faithfulness_rate"),
    ("Private-first routing accuracy", "routing", "private_first_routing_accuracy"),
    ("Unnecessary external searches", "routing", "unnecessary_external_search_rate"),
    ("External fallback accuracy", "routing", "external_fallback_accuracy"),
    ("Query-rewrite success", "routing", "query_rewrite_success_rate"),
    ("Evidence-grading accuracy", "routing", "evidence_grading_accuracy"),
    ("Final-validation failures", "routing", "final_validation_failure_rate"),
    ("Error rate", "system", "error_rate"),
)
RETRIEVAL_KEYS = ("doc_recall@5", "doc_mrr", "chunk_precision@5", "chunk_ndcg@5")
# Settings that only shape the interval calculation, not the answers, so they do not stop two runs from being pooled.
STATISTICS_ONLY_SETTINGS = frozenset({"eval_bootstrap_resamples", "eval_bootstrap_seed", "eval_confidence"})

Counts = dict[tuple[str, str], tuple[int, int]]


@dataclass
class SystemView:
    """Counts and totals for one system (baseline, grade rung or agentic) in one run, or pooled over runs."""

    rates: Counts = field(default_factory=dict)
    n_cases: int = 0
    lat_avg_ms: float | None = None
    lat_p95_ms: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class RunView:
    """What the tables need from one run's ``summary.json``."""

    run_id: str
    status: str
    started_at: str
    settings: dict[str, Any]
    judge_enabled: bool
    n_questions: int
    dataset_name: str
    dataset_sha256: str
    split: str
    units_total: int
    units_recorded: int
    units_ok: int
    systems: dict[str, SystemView]
    retrieval: dict[str, Any] = field(default_factory=dict)
    paired: list[dict[str, Any]] = field(default_factory=list)
    pooled_from: list[str] = field(default_factory=list)
    case_ids: frozenset[str] | None = None  # question ids actually run (from cases.jsonl); None when the file is missing


# ------------------------------------------------------------------------------ loading
def _find_summary(run: str | Path, results_dir: Path) -> Path:
    p = Path(run)
    candidates = [p, p / "summary.json", results_dir / str(run) / "summary.json"]
    for c in candidates:
        if c.is_file() and c.name == "summary.json":
            return c
    raise FileNotFoundError(f"no summary.json for run {run!r} (looked in {', '.join(str(c) for c in candidates)})")


def load_summary(run: str | Path, results_dir: Path) -> dict[str, Any]:
    """Read a run's ``summary.json``; ``run`` is a run id under ``results_dir``, a run folder or the file itself."""
    return json.loads(_find_summary(run, results_dir).read_text(encoding="utf-8"))


def read_case_ids(summary_path: Path) -> frozenset[str] | None:
    """Question ids recorded in the run's ``cases.jsonl`` (None when that file is absent or unreadable)."""
    f = summary_path.with_name("cases.jsonl")
    if not f.is_file():
        return None
    try:
        return frozenset(json.loads(line)["case_id"] for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    except (ValueError, KeyError, TypeError):
        return None


def read_case_outcomes(summary_path: Path) -> dict[str, list[tuple[str, bool]]] | None:
    """``{system: [(category, correct), ...]}`` from the run's ``cases.jsonl`` (None when absent or unreadable); unscored rows are left out."""
    f = summary_path.with_name("cases.jsonl")
    if not f.is_file():
        return None
    out: dict[str, list[tuple[str, bool]]] = {}
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                correct = (row.get("scores") or {}).get("correct")
                if isinstance(correct, bool):
                    out.setdefault(row["mode"], []).append((str(row["category"]), correct))
    except (ValueError, KeyError, TypeError):
        return None
    return out


def add_without_conflicting_rate(view: RunView, outcomes: dict[str, list[tuple[str, bool]]] | None) -> None:
    """Derive "overall correct without the conflicting category" per system from the recorded case outcomes.

    Skipped for a system unless every one of its units was scored, so the counts always add up to the summary's overall rate.
    """
    for mode, system in view.systems.items():
        overall = system.rates.get(("answer", "overall_correct_rate"))
        rows = (outcomes or {}).get(mode)
        if overall is None or rows is None or len(rows) != overall[1]:
            continue
        excluded = [ok for category, ok in rows if category == CONFLICTING_CATEGORY]
        system.rates[WITHOUT_CONFLICTING] = (overall[0] - sum(excluded), overall[1] - len(excluded))


def load_run_view(run: str | Path, results_dir: Path) -> RunView:
    """Summary plus the ids and outcomes of the questions that were run, ready for rendering or pooling."""
    path = _find_summary(run, results_dir)
    view = run_view(json.loads(path.read_text(encoding="utf-8")))
    view.case_ids = read_case_ids(path)
    add_without_conflicting_rate(view, read_case_outcomes(path))
    return view


def _system_view(block: dict[str, Any]) -> SystemView:
    rates: Counts = {}
    for section in ("answer", "routing", "system"):
        for key, v in (block.get(section) or {}).items():
            if isinstance(v, dict) and "num" in v and "den" in v:
                rates[(section, key)] = (int(v["num"]), int(v["den"]))
    s = block.get("system") or {}
    return SystemView(
        rates=rates, n_cases=int(s.get("n_cases") or 0), lat_avg_ms=s.get("latency_avg_ms"), lat_p95_ms=s.get("latency_p95_ms"),
        tokens=int(s.get("total_tokens") or 0), cost_usd=float(s.get("estimated_cost_usd") or 0.0),
    )


def run_view(summary: dict[str, Any]) -> RunView:
    """Normalise a summary. Raises ``ValueError`` when it has no per-system results at all."""
    modes = {m: b for m, b in (summary.get("modes") or {}).items() if b}
    if not modes:
        raise ValueError(f"run {summary.get('run_id')!r} has no per-system results")
    ordered = [m for m in MODE_ORDER if m in modes] + [m for m in modes if m not in MODE_ORDER]
    ds, prog, opts = summary.get("dataset") or {}, summary.get("progress") or {}, summary.get("options") or {}
    return RunView(
        run_id=str(summary.get("run_id", "?")), status=str(summary.get("status", "?")), started_at=str(summary.get("started_at", ""))[:10],
        settings=dict(summary.get("settings") or {}), judge_enabled=bool((summary.get("judge") or {}).get("enabled")),
        n_questions=int(ds.get("num_cases") or 0), dataset_name=Path(str(ds.get("path", "?")).replace("\\", "/")).name,
        dataset_sha256=str(ds.get("sha256", "")), split=str(ds.get("split") or opts.get("split") or "not recorded"),
        units_total=int(prog.get("units_total") or 0), units_recorded=int(prog.get("units_recorded") or 0), units_ok=int(prog.get("units_ok") or 0),
        systems={m: _system_view(modes[m]) for m in ordered}, retrieval=dict(summary.get("retrieval") or {}),
        paired=list((summary.get("intervals") or {}).get("paired") or []),
    )


def check_finished(run: RunView, *, allow_partial: bool) -> list[str]:
    """Warnings for a run that is renderable; raises ``ValueError`` for an unfinished one unless ``allow_partial``."""
    warnings: list[str] = []
    if run.status != "completed" or run.units_recorded < run.units_total:
        msg = f"run {run.run_id} is not complete (status {run.status}, {run.units_recorded}/{run.units_total} units recorded)"
        if not allow_partial:
            raise ValueError(msg + "; resume it first or pass allow_partial")
        warnings.append("PARTIAL RUN: " + msg)
    if run.units_ok < run.units_recorded:
        warnings.append(f"{run.units_recorded - run.units_ok} of {run.units_recorded} recorded units ended in an infrastructure error and are not answers.")
    return warnings


# ------------------------------------------------------------------------------ pooling
def pool(runs: list[RunView]) -> RunView:
    """Add the counts of several runs of the same code and settings. Refuses runs that are not comparable."""
    if len(runs) < 2:
        raise ValueError("pooling needs at least two runs")
    first = runs[0]
    seen: set[tuple[str, str]] = set()
    for i, r in enumerate(runs):
        changed = sorted(k for k in set(r.settings) | set(first.settings) if k not in STATISTICS_ONLY_SETTINGS and r.settings.get(k) != first.settings.get(k))
        if changed:
            raise ValueError(f"cannot pool {first.run_id} with {r.run_id}: settings differ ({', '.join(changed)})")
        if set(r.systems) != set(first.systems) or r.judge_enabled != first.judge_enabled:
            raise ValueError(f"cannot pool {first.run_id} with {r.run_id}: different systems or judge setting")
        for other in runs[:i]:
            shared = sorted((r.case_ids or frozenset()) & (other.case_ids or frozenset()))
            if shared:
                raise ValueError(f"cannot pool {other.run_id} with {r.run_id}: both ran question(s) {', '.join(shared[:5])}{' ...' if len(shared) > 5 else ''}, which would be counted twice")
        key = (r.dataset_sha256, r.split)
        if key in seen:
            raise ValueError(f"cannot pool: {r.run_id} covers the same dataset and split as another run, so its questions would be counted twice")
        seen.add(key)
    systems: dict[str, SystemView] = {}
    for mode in first.systems:
        views = [r.systems[mode] for r in runs]
        total = sum(v.n_cases for v in views)
        keys = {k for v in views for k in v.rates if k[1] not in DERIVED_KEYS or all(k in w.rates for w in views)}  # a derived rate needs every run
        lat = [v for v in views if v.lat_avg_ms is not None]
        systems[mode] = SystemView(
            rates={k: (sum(v.rates.get(k, (0, 0))[0] for v in views), sum(v.rates.get(k, (0, 0))[1] for v in views)) for k in keys},
            n_cases=total, lat_avg_ms=(sum(v.lat_avg_ms * v.n_cases for v in lat) / sum(v.n_cases for v in lat)) if lat and sum(v.n_cases for v in lat) else None,
            lat_p95_ms=None, tokens=sum(v.tokens for v in views), cost_usd=sum(v.cost_usd for v in views),
        )
    return RunView(
        run_id=" + ".join(r.run_id for r in runs), status="pooled", started_at=min(r.started_at for r in runs), settings=first.settings,
        judge_enabled=first.judge_enabled, n_questions=sum(r.n_questions for r in runs), dataset_name=" + ".join(r.dataset_name for r in runs),
        dataset_sha256="", split=" + ".join(r.split for r in runs), units_total=sum(r.units_total for r in runs),
        units_recorded=sum(r.units_recorded for r in runs), units_ok=sum(r.units_ok for r in runs), systems=systems,
        pooled_from=[r.run_id for r in runs], case_ids=frozenset().union(*(r.case_ids or frozenset() for r in runs)) or None,
    )


# ------------------------------------------------------------------------------ formatting
def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _rate_cell(v: SystemView, section: str, key: str) -> str:
    c = v.rates.get((section, key))
    if not c or c[1] == 0:
        return NA
    return f"{_pct(c[0] / c[1])} ({c[0]}/{c[1]})"


def _interval_cell(v: SystemView, section: str, key: str) -> str:
    c = v.rates.get((section, key))
    iv = wilson_interval(c[0], c[1], CONFIDENCE) if c else None
    return NA if iv is None else f"{_pct(c[0] / c[1])} [{_pct(iv[0])} to {_pct(iv[1])}]"


def _pts(x: float) -> str:
    """Signed percentage points, with a plain 0 when the difference rounds to nothing."""
    n = round(x * 100)
    return "0" if n == 0 else f"{n:+d}"


def _seconds(ms: float | None) -> str:
    return NA if ms is None else f"{ms / 1000:.1f} s"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(r) + " |" for r in rows]


def _label(mode: str) -> str:
    return MODE_LABEL.get(mode, mode)


def _shown_rows(run: RunView) -> list[tuple[str, str, str]]:
    """RATE_ROWS, leaving out a derived row when no system has a different count from the row it derives from (nothing was excluded)."""
    def excluded_something(v: SystemView) -> bool:
        derived = v.rates.get(WITHOUT_CONFLICTING)
        return derived is not None and derived != v.rates.get(("answer", "overall_correct_rate"))

    keep = any(excluded_something(v) for v in run.systems.values())
    return [r for r in RATE_ROWS if r[2] not in DERIVED_KEYS or keep]


def render_run(run: RunView, warnings: list[str] | None = None) -> str:
    """Markdown for one run (or one pooled set): facts, caveats, the results table and the interval table."""
    modes = list(run.systems)
    s = run.settings
    shown = _shown_rows(run)
    out: list[str] = [f"### {'Pooled runs' if run.pooled_from else 'Run'} `{run.run_id}`", ""]
    facts = [f"model `{s.get('llm_model', '?')}`", f"reranker {'on' if s.get('rerank_enabled') else 'off'}", f"LLM judge {'on' if run.judge_enabled else 'off'}",
             f"top-k {s.get('retrieval_top_k', '?')}", f"minimum retrieval score {s.get('retrieval_min_score', '?')}", f"{run.n_questions} questions per system",
             f"{run.units_ok} of {run.units_recorded} system-question units answered without an infrastructure error"]
    where = "datasets " if run.pooled_from else "dataset "
    out += [f"Started {run.started_at}; " + ", ".join(facts) + f"; {where}`{run.dataset_name}` (split: {run.split}).", ""]
    notes = list(warnings or [])
    if run.n_questions < SMALL_SAMPLE:
        notes.append(f"Only {run.n_questions} questions per system, below the {SMALL_SAMPLE}-question threshold: differences of a few points are noise and no winner is declared.")
    if not run.judge_enabled:
        notes.append("The LLM judge was off, so faithfulness and answer relevance were not measured; do not read the n/a faithfulness row as a pass.")
    if run.pooled_from:
        notes.append("Pooled by adding the counts behind each rate; P95 latency cannot be pooled. The runs cover different questions, so this is not a repeat of one experiment.")
    if len(shown) == len(RATE_ROWS):
        notes.append("The row without the conflicting category is shown because its labels (C01-C06, F29-F31) expect the older edition of the password policy, which is now marked superseded: a system that states only the current rule scores incorrect on them (docs/FINDINGS.md, finding 10). No claim about conflicting documents is made from this run.")
    notes.append("Intervals reflect which questions were asked, not LLM sampling noise between repeated runs.")
    out += [f"> {n}" for n in notes] + [""]

    rows = [[label] + [_rate_cell(run.systems[m], sec, key) for m in modes] for label, sec, key in shown]
    rows.append(["Average / P95 latency"] + [f"{_seconds(run.systems[m].lat_avg_ms)} / {_seconds(run.systems[m].lat_p95_ms)}" for m in modes])
    rows.append(["Total tokens"] + [f"{run.systems[m].tokens:,}" for m in modes])
    rows.append(["Approx. cost (USD)"] + [f"{run.systems[m].cost_usd:.3f}" for m in modes])
    out += _table(["Metric"] + [_label(m) for m in modes], rows) + ["", "**95% Wilson intervals for the same rates** (from the counts above)", ""]
    out += _table(["Metric"] + [_label(m) for m in modes], [[label] + [_interval_cell(run.systems[m], sec, key) for m in modes] for label, sec, key in shown]) + [""]

    ret = (run.retrieval or {}).get("metrics") or {}
    if ret and not run.pooled_from:
        shown = ", ".join(f"{k.replace('_', ' ')} {ret[k]:.2f}" for k in RETRIEVAL_KEYS if isinstance(ret.get(k), (int, float)))
        out += [f"Retrieval ({run.retrieval.get('n_cases', '?')} questions with expected documents): {shown}.", ""]
    for p in run.paired:
        if run.pooled_from or not p.get("metrics"):
            continue
        out += [f"Paired difference, {p.get('from')} to {p.get('to')} (same {p.get('n_cases')} questions, bootstrap interval):", ""]
        out += _table(["Metric", "Difference", "95% interval", "Verdict"], [
            [m["label"], f"{_pts(m['diff'])} pts", f"{_pts(m['lo'])} to {_pts(m['hi'])} pts", m["verdict"]] for m in p["metrics"]]) + [""]
    return "\n".join(out)


def render_many(runs: list[RunView], *, pooled: bool, allow_partial: bool = False) -> str:
    """Markdown for each run in turn, followed by the pooled table when ``pooled`` (raises for an unfinished run)."""
    warnings = [check_finished(r, allow_partial=allow_partial) for r in runs]
    parts = [render_run(r, w) for r, w in zip(runs, warnings)]
    if pooled:
        parts.append(render_run(pool(runs), [w for ws in warnings for w in ws]))
    return "\n".join(parts)
