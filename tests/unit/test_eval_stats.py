"""Confidence intervals for the evaluation: Wilson rates, paired bootstrap differences, and the per-run ``intervals`` block."""
from __future__ import annotations

import pytest

from copilot.config import PROJECT_ROOT
from copilot.evaluation.paired import (
    MIN_CASES_FOR_VERDICT, build_intervals, comparison_pairs, paired_comparison, rate_intervals,
)
from copilot.evaluation.reporting import flatten_metrics
from copilot.evaluation.results import CaseResult
from copilot.evaluation.runner import RunOptions
from copilot.evaluation.schemas import load_dataset
from copilot.evaluation.stats import paired_bootstrap, wilson_interval
from eval_helpers import StubAgent, grade_rung_stub


# ------------------------------------------------------------------------------ Wilson interval
@pytest.mark.parametrize("num,den,lo,hi", [
    (0, 10, 0.0, 0.27753), (10, 10, 0.72247, 1.0), (5, 10, 0.23659, 0.76341), (51, 93, 0.44733, 0.64560),
])
def test_wilson_matches_known_values(num, den, lo, hi):
    got = wilson_interval(num, den)
    assert got[0] == pytest.approx(lo, abs=1e-4) and got[1] == pytest.approx(hi, abs=1e-4)


def test_wilson_edges_widening_and_validation():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0       # exact bounds at the extremes
    narrow, wide = wilson_interval(30, 93, 0.90), wilson_interval(30, 93, 0.99)
    assert wide[0] < narrow[0] < 30 / 93 < narrow[1] < wide[1]
    lo_small, hi_small = wilson_interval(5, 10)
    lo_big, hi_big = wilson_interval(50, 100)
    assert (hi_big - lo_big) < (hi_small - lo_small)                                    # more questions, tighter interval
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(5, 10, confidence=0.3)


# ------------------------------------------------------------------------------ paired bootstrap
def test_paired_bootstrap_clear_improvement_no_change_and_swap():
    better = paired_bootstrap([0] * 30, [1] * 30)
    assert better == {"n": 30, "diff": 1.0, "lo": 1.0, "hi": 1.0, "excludes_zero": True, "gained": 30, "lost": 0}
    same = paired_bootstrap([1, 0, 1, 1] * 5, [1, 0, 1, 1] * 5)
    assert same["diff"] == 0.0 and not same["excludes_zero"] and same["gained"] == same["lost"] == 0

    before = [0] * 10 + [1] * 2 + [1] * 14 + [0] * 14          # 10 questions gained, 2 lost, 28 unchanged
    after = [1] * 10 + [0] * 2 + [1] * 14 + [0] * 14
    r = paired_bootstrap(before, after)
    assert (r["n"], r["gained"], r["lost"]) == (40, 10, 2) and r["diff"] == pytest.approx(0.2)
    assert 0.0 < r["lo"] < 0.2 < r["hi"] < 0.5 and r["excludes_zero"]
    back = paired_bootstrap(after, before)
    assert back["diff"] == pytest.approx(-0.2) and (back["gained"], back["lost"]) == (2, 10) and back["hi"] < 0


def test_paired_bootstrap_small_net_change_is_not_significant_and_is_reproducible():
    before = [0] * 3 + [1] + [0] * 30 + [1] * 59
    after = [1] * 3 + [0] + [0] * 30 + [1] * 59                # 3 gained, 1 lost of 93
    r = paired_bootstrap(before, after)
    assert r["diff"] == pytest.approx(2 / 93) and r["lo"] < 0 < r["hi"] and not r["excludes_zero"]
    assert paired_bootstrap(before, after) == r and paired_bootstrap(before, after, seed=1)["diff"] == r["diff"]


def test_paired_bootstrap_rejects_bad_input():
    for args in (([1], [1, 0]), ([], [])):
        with pytest.raises(ValueError):
            paired_bootstrap(*args)
    with pytest.raises(ValueError):
        paired_bootstrap([1], [0], resamples=0)


# ------------------------------------------------------------------------------ paired comparison on the real dataset
@pytest.fixture(scope="module")
def cases():
    return load_dataset(PROJECT_ROOT / "data" / "eval" / "dataset.json")


def _result(case, mode, correct):
    abstain = case.expected_abstention
    outcome = ("correct_abstention" if correct else "hallucinated_answer") if abstain else ("correct" if correct else "false_abstention")
    return CaseResult(
        case_id=case.id, category=case.category, mode=mode, question=case.question, abstained=(correct if abstain else not correct),
        scores={"correct": correct, "outcome": outcome, "citation_correct": None if abstain else correct},
    )


def _systems(cases):
    """'before' misses every third question; 'after' answers all of them correctly."""
    before = [_result(c, "baseline", i % 3 != 0) for i, c in enumerate(cases)]
    after = [_result(c, "agentic", True) for c in cases]
    return before, after


def _by_key(comparison):
    return {m["key"]: m for m in comparison["metrics"]}


def test_paired_comparison_counts_flipped_questions_and_calls_a_clear_gain_better(cases):
    before, after = _systems(cases)
    cmp = paired_comparison(cases, before, after, resamples=1000, seed=3, confidence=0.95)
    m = _by_key(cmp)
    assert cmp["n_cases"] == len(cases) == 93
    assert (m["overall_correct"]["gained"], m["overall_correct"]["lost"], m["overall_correct"]["n"]) == (31, 0, 93)
    assert m["overall_correct"]["before"] == pytest.approx(62 / 93) and m["overall_correct"]["after"] == 1.0
    assert m["overall_correct"]["verdict"] == "better" and m["overall_correct"]["lo"] > 0
    assert m["hallucinated"]["n"] == 30 and m["hallucinated"]["after"] == 0.0 and m["hallucinated"]["diff"] < 0
    assert m["hallucinated"]["verdict"] == "better"                                     # lower is better, and it fell
    assert m["answer_correct"]["n"] == 63 and m["correct_abstention"]["n"] == 30        # each metric uses only the questions it applies to


def test_paired_comparison_reverse_direction_is_worse_and_small_samples_are_not_judged(cases):
    before, after = _systems(cases)
    reverse = _by_key(paired_comparison(cases, after, before, resamples=1000, seed=3, confidence=0.95))
    assert reverse["overall_correct"]["verdict"] == "worse" and reverse["overall_correct"]["lost"] == 31
    assert reverse["hallucinated"]["verdict"] == "worse"

    few = _by_key(paired_comparison(cases[:5], before[:5], after[:5], resamples=1000, seed=3, confidence=0.95))
    assert few["overall_correct"]["n"] == 5 < MIN_CASES_FOR_VERDICT and few["overall_correct"]["verdict"] == "too few questions"


def test_paired_comparison_uses_only_questions_both_systems_answered(cases):
    before, after = _systems(cases)
    cmp = paired_comparison(cases, before[:40], after[20:60], resamples=500, seed=1, confidence=0.9)
    assert cmp["n_cases"] == 20 and _by_key(cmp)["overall_correct"]["n"] == 20


def test_rate_intervals_use_the_same_definitions_as_the_summary(cases):
    before, _ = _systems(cases)
    r = rate_intervals(cases, before, 0.95)
    assert r["overall_correct"]["num"] == 62 and r["overall_correct"]["den"] == 93
    assert r["overall_correct"]["lo"] < 62 / 93 < r["overall_correct"]["hi"]
    assert r["hallucinated"]["den"] == 30 and r["answer_correct"]["den"] == 63


def test_comparison_pairs_follow_the_ladder_and_always_include_baseline_vs_agentic():
    assert comparison_pairs(["baseline", "agentic"], False) == [("baseline", "agentic")]
    assert comparison_pairs(["agentic"], False) == [] and comparison_pairs(["baseline"], True) == []
    assert comparison_pairs(["baseline", "grade", "agentic"], True) == [("baseline", "grade"), ("grade", "agentic"), ("baseline", "agentic")]
    assert comparison_pairs(["grade", "agentic"], True) == [("grade", "agentic")]                 # no baseline, no extra pair


def test_build_intervals_block(cases, settings):
    before, after = _systems(cases)
    iv = build_intervals(cases, {"baseline": before, "agentic": after}, ["baseline", "agentic"], settings, is_ablation=False)
    assert iv["confidence"] == 0.95 and iv["resamples"] == settings.eval_bootstrap_resamples and "LLM sampling" in iv["caveat"]
    assert set(iv["rates"]) == {"baseline", "agentic"} and [(p["from"], p["to"]) for p in iv["paired"]] == [("baseline", "agentic")]
    assert iv == build_intervals(cases, {"baseline": before, "agentic": after}, ["baseline", "agentic"], settings, is_ablation=False)   # seeded
    solo = build_intervals(cases, {"agentic": after}, ["agentic"], settings, is_ablation=False)
    assert solo["paired"] == [] and "agentic" in solo["rates"]
    assert build_intervals(cases, {"baseline": []}, ["baseline"], settings, is_ablation=False) is None


# ------------------------------------------------------------------------------ in a real runner run
def test_runner_summary_reports_intervals_and_exports_them(make_runner):
    runner, *_ = make_runner(agent_factory=lambda overrides: StubAgent(grade_rung_stub))
    s = runner.run(RunOptions(variants=["baseline", "grade", "agentic"], run_id="ci-abl", use_judge=False))
    iv = s["intervals"]
    assert [(p["from"], p["to"]) for p in iv["paired"]] == [("baseline", "grade"), ("grade", "agentic"), ("baseline", "agentic")]
    assert iv["rates"]["agentic"]["overall_correct"]["den"] == 3 and iv["rates"]["agentic"]["overall_correct"]["value"] == 1.0
    assert {m["verdict"] for p in iv["paired"] for m in p["metrics"]} == {"too few questions"}     # three questions can prove nothing
    rows = flatten_metrics(s)
    assert any(r["scope"] == "interval.agentic" for r in rows) and any(r["scope"] == "paired.grade->agentic" for r in rows)
    assert (runner.run_dir("ci-abl") / "metrics.csv").read_text().count("paired.") >= 1

    classic, *_ = make_runner()
    two = classic.run(RunOptions(run_id="ci-classic", use_judge=False))
    assert [(p["from"], p["to"]) for p in two["intervals"]["paired"]] == [("baseline", "agentic")]
