"""Ablation study plumbing: the variant ladder, run options, the runner's per-variant agents and the step table. No network."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from copilot.config import Settings
from copilot.evaluation import adapters, variants
from copilot.evaluation.aggregate import ablation_table, rate
from copilot.evaluation.reporting import flatten_metrics
from copilot.evaluation.runner import RunOptions
from copilot.evaluation.schemas import load_dataset
from copilot.evaluation.scoring import score_case
from copilot.evaluation.store import EvalRunStore
from eval_helpers import VPN_Q, StubAgent, agentic_response, grade_rung_stub, make_answer, write_dataset


# ------------------------------------------------------------------------------ the ladder itself
def test_ladder_is_ordered_and_ends_with_the_real_application():
    assert variants.NAMES == ("baseline", "classify", "grade", "rewrite", "external", "agentic")
    assert variants.get_variant("baseline").kind == "baseline" and not variants.is_agent("baseline")
    assert all(variants.is_agent(n) for n in variants.NAMES[1:])
    assert variants.overrides_for("baseline") == {} and variants.overrides_for("agentic") == {}   # the real app is not modified


def test_every_override_is_a_real_setting():
    """A typo in an override would silently change nothing (model_copy does not validate), so check the names."""
    for v in variants.LADDER:
        assert set(v.overrides) <= set(Settings.model_fields), v.name


def test_each_rung_removes_one_restriction_of_the_one_before():
    """Cumulative ladder: a rung's restrictions are a subset of the previous rung's, and exactly one thing is lifted per step."""
    agents = [variants.get_variant(n) for n in variants.NAMES[1:]]
    for prev, cur in zip(agents, agents[1:]):
        assert set(cur.overrides) < set(prev.overrides), (prev.name, cur.name)
        assert len(set(prev.overrides) - set(cur.overrides)) == 1, (prev.name, cur.name)
    assert agents[0].overrides["agent_use_grader"] is False and "agent_use_grader" not in agents[1].overrides


def test_normalise_and_parse_validate_and_order_names():
    assert variants.normalise(["agentic", "baseline", "baseline", " grade "]) == ["baseline", "grade", "agentic"]
    assert variants.parse_variants("agentic, baseline") == ["baseline", "agentic"]
    assert variants.parse_variants("ALL") == list(variants.NAMES)
    for bad in ("nonsense", "baseline,nonsense"):
        with pytest.raises(ValueError, match="unknown variant"):
            variants.parse_variants(bad)
    with pytest.raises(ValueError, match="no variants"):
        variants.parse_variants(" , ")


# ------------------------------------------------------------------------------ run options
def test_run_options_order_validate_and_label_ablation_runs():
    o = RunOptions(variants=["agentic", "grade", "baseline"])
    assert o.variants == ["baseline", "grade", "agentic"] and o.modes == ["baseline", "grade", "agentic"]
    assert RunOptions().modes == ["baseline", "agentic"] and RunOptions().variants is None          # unchanged default
    with pytest.raises(ValidationError, match="either"):
        RunOptions(mode="agentic", variants=["grade"])
    with pytest.raises(ValidationError, match="unknown variant"):
        RunOptions(variants=["warp"])


def test_store_mode_always_satisfies_the_database_constraint():
    allowed = {"agentic", "baseline", "both"}
    for opts, expected in [
        (RunOptions(), "both"), (RunOptions(mode="agentic"), "agentic"), (RunOptions(mode="baseline"), "baseline"),
        (RunOptions(variants=["baseline", "grade"]), "both"), (RunOptions(variants=["grade", "agentic"]), "agentic"),
        (RunOptions(variants=["baseline"]), "baseline"),
    ]:
        assert opts.store_mode == expected and opts.store_mode in allowed


# ------------------------------------------------------------------------------ adapters and scoring for variants
def test_a_skipped_grader_is_not_scored_as_a_grading_decision(tmp_path):
    d1 = load_dataset(write_dataset(tmp_path / "c.json"))[0]
    skipped = {"effective_sufficient": True, "confidence": 1.0, "skipped": True}
    r, _, _ = adapters.from_agentic(d1, agentic_response(VPN_Q, make_answer(), private_grade=skipped), "classify")
    assert r.mode == "classify" and r.grade_sufficient is None and r.grade_confidence is None
    scores = score_case(d1, r)
    assert scores["private_first_ok"] is True and scores["grading_correct"] is None     # routing is scored; the missing grader is not
    full, _, _ = adapters.from_agentic(d1, agentic_response(VPN_Q, make_answer()))
    assert full.mode == "agentic" and full.grade_sufficient is True                       # the default is unchanged


# ------------------------------------------------------------------------------ the runner
def test_ablation_run_builds_one_agent_per_rung_and_tabulates_each_step(make_runner):
    received: list[dict] = []

    def factory(overrides):
        received.append(overrides)
        return StubAgent(grade_rung_stub)

    runner, full_agent, baseline, _ = make_runner(agent_factory=factory)
    summary = runner.run(RunOptions(variants=["baseline", "grade", "agentic"], run_id="abl", use_judge=False))

    assert summary["status"] == "completed" and list(summary["modes"]) == ["baseline", "grade", "agentic"]
    assert received == [variants.overrides_for("grade")]                       # built once, with exactly that rung's overrides
    assert len(full_agent.calls) == 3 and len(baseline.calls) == 3             # the injected agent IS the full rung
    assert summary["variants"]["grade"] == variants.overrides_for("grade") and summary["variants"]["agentic"] == {}
    assert summary["progress"] == {"units_total": 9, "units_recorded": 9, "units_ok": 9}

    overall = {m: summary["modes"][m]["answer"]["overall_correct_rate"]["value"] for m in summary["modes"]}
    assert overall == {"baseline": pytest.approx(1 / 3), "grade": pytest.approx(2 / 3), "agentic": 1.0}
    assert summary["modes"]["grade"]["routing"]["private_first_routing_accuracy"]["value"] == 1.0   # variants get routing metrics too
    assert summary["comparison"] is not None                                  # baseline vs the full agent is still reported

    table = summary["ablation"]
    assert table["order"] == ["baseline", "grade", "agentic"] and [r["label"] for r in table["rows"]][0] == "Baseline RAG"
    assert [(s["from"], s["to"]) for s in table["steps"]] == [("baseline", "grade"), ("grade", "agentic")]
    for step in table["steps"]:
        assert next(c for c in step["changes"] if c["key"] == "overall_correct_rate")["verdict"] == "better"
    assert any("Each rung is one run" in line for line in table["findings"])

    run_dir = runner.run_dir("abl")
    modes_on_disk = {json.loads(line)["mode"] for line in (run_dir / "cases.jsonl").read_text().splitlines()}
    assert modes_on_disk == {"baseline", "grade", "agentic"}
    assert json.loads((run_dir / "config.json").read_text())["variants"]["grade"] == variants.overrides_for("grade")
    assert any(r["scope"] == "ablation.baseline->grade" for r in flatten_metrics(summary))
    row = EvalRunStore(runner._s.sqlite_path).get("abl")
    assert row["mode"] == "both" and row["metadata"]["variants"] == ["baseline", "grade", "agentic"]


def test_ablation_without_an_agent_factory_fails_before_spending_anything(make_runner):
    runner, agent, baseline, _ = make_runner()
    with pytest.raises(ValueError, match="agent factory"):
        runner.run(RunOptions(variants=["grade"], run_id="nofactory"))
    assert not agent.calls and not baseline.calls
    # the two classic modes need no factory
    assert runner.run(RunOptions(mode="agentic", use_judge=False, run_id="classic"))["status"] == "completed"


def test_ablation_run_can_be_resumed_with_the_same_rungs(make_runner):
    runner, *_ = make_runner(agent_factory=lambda overrides: StubAgent())
    runner.run(RunOptions(variants=["classify", "agentic"], run_id="abl2", use_judge=False))
    resumed, _ = runner.prepare(RunOptions(run_id="abl2", resume=True))
    assert resumed.variants == ["classify", "agentic"] and resumed.modes == ["classify", "agentic"]


def test_two_system_runs_keep_their_old_shape(make_runner):
    """No ablation block, no variants entry: the existing baseline-vs-agentic output is untouched."""
    runner, *_ = make_runner()
    s = runner.run(RunOptions(run_id="classic2", use_judge=False))
    assert s["ablation"] is None and s["variants"] is None and list(s["modes"]) == ["baseline", "agentic"]


# ------------------------------------------------------------------------------ the step table in isolation
def _mode(correct: tuple[int, int], tokens: int, leaks: int = 0) -> dict:
    return {"answer": {"overall_correct_rate": rate(*correct), "restricted_content_leaks": leaks}, "system": {"n_cases": correct[1], "total_tokens": tokens}}


def test_step_verdicts_use_a_two_point_tolerance_for_rates_and_five_percent_for_counts():
    modes = {"a": _mode((50, 100), 1000), "b": _mode((51, 100), 1040), "c": _mode((60, 100), 2000, leaks=1)}
    table = ablation_table(modes, ["a", "b", "c"], {"a": "A", "b": "B", "c": "C"})

    def verdicts(step):
        return {c["key"]: c["verdict"] for c in step["changes"]}

    assert verdicts(table["steps"][0]) == {"overall_correct_rate": "tie", "restricted_content_leaks": "tie", "total_tokens": "tie"}    # +1 point, +4%
    assert verdicts(table["steps"][1]) == {"overall_correct_rate": "better", "restricted_content_leaks": "worse", "total_tokens": "worse"}
    assert "C: better on 1 (Overall correct); worse on 2" in table["findings"][1]


def test_table_skips_missing_rungs_and_missing_values():
    modes = {"a": _mode((5, 10), 10), "c": _mode((6, 10), 10)}
    table = ablation_table(modes, ["a", "b", "c"], {})
    assert table["order"] == ["a", "c"] and [(s["from"], s["to"]) for s in table["steps"]] == [("a", "c")]
    empty = {"a": _mode((0, 0), 0), "c": _mode((6, 10), 10)}
    assert "overall_correct_rate" not in {c["key"] for c in ablation_table(empty, ["a", "c"], {})["steps"][0]["changes"]}
