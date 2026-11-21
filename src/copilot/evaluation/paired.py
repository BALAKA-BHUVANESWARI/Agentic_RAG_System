"""Uncertainty for a run: an interval for each system's rates and a paired comparison between systems.

Both systems answered the SAME questions, so the honest comparison is per question: how many did the second system
get right that the first got wrong (``gained``), how many the other way round (``lost``), and what range of
differences is compatible with that many questions (:func:`copilot.evaluation.stats.paired_bootstrap`).

A difference is called ``better`` or ``worse`` only when its interval excludes zero and at least
``MIN_CASES_FOR_VERDICT`` questions contribute; otherwise it is ``not distinguishable`` or ``too few questions``
(a bootstrap over three questions would otherwise print a confident-looking interval). That is a statement about which questions happened to be in the dataset, not about repeat
runs of the LLM (see ``stats.py``). The tolerance-based ``tie`` / ``better`` verdicts in ``aggregate.py`` remain as a
quick display rule; this module is the stronger evidence.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from copilot.config import Settings
from copilot.evaluation.results import CaseResult
from copilot.evaluation.schemas import EvalCase
from copilot.evaluation.stats import paired_bootstrap, wilson_interval

Outcome = Callable[[EvalCase, CaseResult], "bool | None"]  # None: the metric does not apply to this question


def _overall(c: EvalCase, r: CaseResult) -> bool:
    return bool(r.scores.get("correct"))


def _answer_correct(c: EvalCase, r: CaseResult) -> bool | None:
    return None if c.expected_abstention else bool(r.scores.get("correct"))


def _correct_abstention(c: EvalCase, r: CaseResult) -> bool | None:
    return bool(r.scores.get("correct")) if c.expected_abstention else None


def _hallucinated(c: EvalCase, r: CaseResult) -> bool | None:
    return (r.scores.get("outcome") == "hallucinated_answer") if c.expected_abstention else None


def _false_abstention(c: EvalCase, r: CaseResult) -> bool | None:
    return None if c.expected_abstention else bool(r.abstained)


def _citation_correct(c: EvalCase, r: CaseResult) -> bool | None:
    v = r.scores.get("citation_correct")
    return None if v is None else bool(v)


# key, label, higher_is_better, per-question outcome. These mirror the definitions in ``aggregate.summarize_mode``.
PAIRED_METRICS: list[tuple[str, str, bool, Outcome]] = [
    ("overall_correct", "Overall correct", True, _overall),
    ("answer_correct", "Answerable questions correct", True, _answer_correct),
    ("correct_abstention", "Correct abstentions", True, _correct_abstention),
    ("hallucinated", "Hallucinated answers", False, _hallucinated),
    ("false_abstention", "False abstentions", False, _false_abstention),
    ("citation_correct", "Citations correct", True, _citation_correct),
]


MIN_CASES_FOR_VERDICT = 10


def _by_case(results: Sequence[CaseResult]) -> dict[str, CaseResult]:
    return {r.case_id: r for r in results}


def rate_intervals(cases: list[EvalCase], results: Sequence[CaseResult], confidence: float) -> dict[str, Any]:
    """Wilson interval for each headline rate of one system."""
    by_id = {c.id: c for c in cases}
    out: dict[str, Any] = {}
    for key, label, _, fn in PAIRED_METRICS:
        flags = [v for r in results if r.case_id in by_id and (v := fn(by_id[r.case_id], r)) is not None]
        if not flags:
            continue
        num, den = sum(flags), len(flags)
        ci = wilson_interval(num, den, confidence)
        out[key] = {"label": label, "value": num / den, "num": num, "den": den, "lo": ci[0] if ci else None, "hi": ci[1] if ci else None}
    return out


def paired_comparison(
    cases: list[EvalCase], before: Sequence[CaseResult], after: Sequence[CaseResult], *, resamples: int, seed: int, confidence: float,
) -> dict[str, Any]:
    """Per-metric paired difference ``after - before`` over the questions both systems answered."""
    by_id = {c.id: c for c in cases}
    a, b = _by_case(before), _by_case(after)
    common = sorted(set(a) & set(b) & set(by_id))
    metrics = []
    for key, label, higher, fn in PAIRED_METRICS:
        xs, ys = [], []
        for cid in common:
            x, y = fn(by_id[cid], a[cid]), fn(by_id[cid], b[cid])
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))
        if not xs:
            continue
        stat = paired_bootstrap(xs, ys, resamples=resamples, seed=seed, confidence=confidence)
        if stat["n"] < MIN_CASES_FOR_VERDICT:
            verdict = "too few questions"
        elif not stat["excludes_zero"]:
            verdict = "not distinguishable"
        else:
            verdict = "better" if (stat["diff"] > 0) == higher else "worse"
        metrics.append({
            "key": key, "label": label, "n": stat["n"], "before": sum(xs) / len(xs), "after": sum(ys) / len(ys),
            "diff": stat["diff"], "lo": stat["lo"], "hi": stat["hi"], "gained": stat["gained"], "lost": stat["lost"], "verdict": verdict,
        })
    return {"n_cases": len(common), "metrics": metrics}


def comparison_pairs(modes: list[str], is_ablation: bool) -> list[tuple[str, str]]:
    """Which (before, after) pairs to compare: consecutive ladder rungs, plus baseline -> agentic when both are present."""
    pairs = list(zip(modes, modes[1:])) if is_ablation else []
    if "baseline" in modes and "agentic" in modes and ("baseline", "agentic") not in pairs:
        pairs.append(("baseline", "agentic"))
    return pairs


def build_intervals(cases: list[EvalCase], per_mode: dict[str, list[CaseResult]], modes: list[str], settings: Settings, *, is_ablation: bool) -> dict[str, Any] | None:
    """The ``intervals`` block of ``summary.json`` (None when nothing was answered)."""
    present = [m for m in modes if per_mode.get(m)]
    if not present:
        return None
    conf, resamples, seed = settings.eval_confidence, settings.eval_bootstrap_resamples, settings.eval_bootstrap_seed
    paired = []
    for before, after in comparison_pairs(present, is_ablation):
        cmp = paired_comparison(cases, per_mode[before], per_mode[after], resamples=resamples, seed=seed, confidence=conf)
        paired.append({"from": before, "to": after, **cmp})
    return {
        "confidence": conf, "resamples": resamples, "seed": seed,
        "method": "Wilson score interval per rate; paired percentile bootstrap over questions for differences",
        "caveat": "Intervals reflect which questions were asked, not LLM sampling noise between repeated runs.",
        "rates": {m: rate_intervals(cases, per_mode[m], conf) for m in present},
        "paired": paired,
    }
