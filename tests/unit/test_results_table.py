"""README results tables rendered from saved runs: every number comes from summary.json, unfinished or incomparable runs are refused."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from copilot.evaluation.results_table import (
    SMALL_SAMPLE, CONFLICTING_CATEGORY, WITHOUT_CONFLICTING, check_finished, load_run_view, load_summary, pool, read_case_ids, render_many, render_run, run_view,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_results.py"


def _rate(num: int, den: int) -> dict:
    return {"value": (num / den) if den else None, "num": num, "den": den}


def make_summary(
    run_id: str = "r1", *, n: int = 40, overall=((30, 40), (36, 40)), status: str = "completed", judge: bool = True,
    sha: str = "sha-1", split: str = "dev", settings: dict | None = None, units_ok: int | None = None,
    extra_mode: bool = False, paired: bool = False,
) -> dict:
    """A summary.json shaped like the runner's, with only the keys the renderer reads."""
    def system(tokens: int, cost: float, avg: float, p95: float) -> dict:
        return {"n_cases": n, "latency_avg_ms": avg, "latency_p95_ms": p95, "error_rate": _rate(0, n), "total_tokens": tokens, "estimated_cost_usd": cost}

    baseline = {"answer": {"overall_correct_rate": _rate(*overall[0]), "faithfulness_rate": _rate(0, 0) if not judge else _rate(9, 10)},
                "routing": {}, "system": system(1234, 0.0044, 2156.0, 3931.0)}
    agentic = {"answer": {"overall_correct_rate": _rate(*overall[1]), "faithfulness_rate": _rate(0, 0) if not judge else _rate(10, 10)},
               "routing": {"private_first_routing_accuracy": _rate(19, 20), "unnecessary_external_search_rate": _rate(0, 20)},
               "system": system(56789, 0.0487, 23299.0, 61571.0)}
    modes = {"baseline": baseline, "agentic": agentic}
    if extra_mode:
        modes["grade"] = {"answer": {"overall_correct_rate": _rate(*overall[0])}, "routing": {}, "system": system(2000, 0.01, 5000.0, 9000.0)}
    summary = {
        "run_id": run_id, "status": status, "started_at": "2026-11-23T19:50:00+00:00", "options": {"split": split},
        "dataset": {"path": "C:\\proj\\data\\eval\\dataset.json", "sha256": sha, "num_cases": n, "split": split},
        "settings": {"llm_model": "model-x", "rerank_enabled": False, "retrieval_top_k": 5, "retrieval_min_score": 0.25, "eval_bootstrap_seed": 7, **(settings or {})},
        "progress": {"units_total": 2 * n, "units_recorded": 2 * n if status == "completed" else n, "units_ok": units_ok if units_ok is not None else (2 * n if status == "completed" else n)},
        "judge": {"enabled": judge},
        "retrieval": {"n_cases": 12, "metrics": {"doc_recall@5": 1.0, "doc_mrr": 0.97, "chunk_precision@5": 0.3, "chunk_ndcg@5": 0.87}},
        "modes": modes,
    }
    if paired:
        summary["intervals"] = {"paired": [{"from": "baseline", "to": "agentic", "n_cases": n, "metrics": [
            {"label": "Overall correct", "diff": 0.15, "lo": 0.05, "hi": 0.25, "verdict": "better"},
            {"label": "False abstentions", "diff": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "not distinguishable"}]}]}
    return summary


def view(**kw):
    return run_view(make_summary(**kw))


# ------------------------------------------------------------------------------ cells and tables
def test_table_cells_come_from_the_summary():
    text = render_run(view())
    assert "| Overall correct | 75% (30/40) | 90% (36/40) |" in text
    assert "| Private-first routing accuracy | n/a | 95% (19/20) |" in text           # the baseline has no routing
    assert "| Unnecessary external searches | n/a | 0% (0/20) |" in text
    assert "| Faithful (LLM judge) | 90% (9/10) | 100% (10/10) |" in text
    assert "| Average / P95 latency | 2.2 s / 3.9 s | 23.3 s / 61.6 s |" in text
    assert "| Total tokens | 1,234 | 56,789 |" in text and "| Approx. cost (USD) | 0.004 | 0.049 |" in text
    assert "model `model-x`" in text and "reranker off" in text and "`dataset.json` (split: dev)" in text


def test_interval_table_uses_wilson_counts():
    text = render_run(view(n=10, overall=((5, 10), (10, 10))))
    assert "| Overall correct | 50% [24% to 76%] | 100% [72% to 100%] |" in text         # same values the stats tests pin for 5/10 and 10/10
    assert "Intervals reflect which questions were asked" in text


def test_small_sample_caveat_only_below_the_threshold():
    assert "below the 30-question threshold" in render_run(view(n=SMALL_SAMPLE - 1))
    assert "below the 30-question threshold" not in render_run(view(n=SMALL_SAMPLE))


def test_small_sample_constant_matches_the_dashboard():
    ui = pytest.importorskip("copilot.ui.eval_formatting")
    assert SMALL_SAMPLE == ui.SMALL_SAMPLE


def test_judge_off_is_stated_and_faithfulness_is_not_zero():
    text = render_run(view(judge=False))
    assert "The LLM judge was off" in text
    assert "| Faithful (LLM judge) | n/a | n/a |" in text


def test_ablation_run_gets_a_column_per_system_in_ladder_order():
    text = render_run(view(extra_mode=True))
    header = next(line for line in text.splitlines() if line.startswith("| Metric |"))
    assert header == "| Metric | Baseline RAG | Grader only (ablation rung) | Agentic workflow |"


def test_paired_differences_are_shown_and_zero_is_plain():
    text = render_run(view(paired=True))
    assert "Paired difference, baseline to agentic (same 40 questions" in text
    assert "| Overall correct | +15 pts | +5 to +25 pts | better |" in text
    assert "| False abstentions | 0 pts | 0 to 0 pts | not distinguishable |" in text


def test_retrieval_line_is_printed_for_a_single_run():
    assert "Retrieval (12 questions with expected documents): doc recall@5 1.00, doc mrr 0.97" in render_run(view())


def test_summary_without_results_is_rejected():
    s = make_summary()
    s["modes"] = {"baseline": {}, "agentic": {}}
    with pytest.raises(ValueError, match="no per-system results"):
        run_view(s)


# ------------------------------------------------------------------------------ unfinished and failed runs
def test_unfinished_run_is_refused_unless_allowed():
    v = view(status="stopped")
    with pytest.raises(ValueError, match="not complete"):
        check_finished(v, allow_partial=False)
    with pytest.raises(ValueError, match="not complete"):
        render_many([v], pooled=False)
    warnings = check_finished(v, allow_partial=True)
    assert warnings and warnings[0].startswith("PARTIAL RUN")
    assert "PARTIAL RUN" in render_many([v], pooled=False, allow_partial=True)


def test_infrastructure_errors_are_called_out():
    warnings = check_finished(view(n=10, units_ok=17), allow_partial=False)
    assert warnings == ["3 of 20 recorded units ended in an infrastructure error and are not answers."]


# ------------------------------------------------------------------------------ pooling
def _disjoint(a: str, b: str, **kw):
    va, vb = view(run_id=a, sha=f"sha-{a}", **kw), view(run_id=b, sha=f"sha-{b}", **kw)
    va.case_ids, vb.case_ids = frozenset({"F01", "F02"}), frozenset({"D01", "D02"})
    return va, vb


def test_pool_adds_counts_and_does_not_average_percentages():
    a = run_view(make_summary("a", n=4, overall=((3, 4), (4, 4)), sha="sa", split="fresh"))
    b = run_view(make_summary("b", n=16, overall=((9, 16), (12, 16)), sha="sb", split="test"))
    a.case_ids, b.case_ids = frozenset({"F01"}), frozenset({"D01"})
    pooled = pool([a, b])
    assert pooled.systems["baseline"].rates[("answer", "overall_correct_rate")] == (12, 20)     # 60%; averaging 75% and 56% would give 66%
    text = render_run(pooled)
    assert "| Overall correct | 60% (12/20) | 80% (16/20) |" in text
    assert "| Total tokens | 2,468 | 113,578 |" in text
    assert "| Average / P95 latency | 2.2 s / n/a | 23.3 s / n/a |" in text                      # P95 cannot be pooled
    assert pooled.n_questions == 20 and pooled.case_ids == frozenset({"F01", "D01"})
    assert "Pooled by adding the counts" in text and "Retrieval (" not in text


def test_pool_weights_average_latency_by_question_count():
    a = run_view(make_summary("a", n=10, sha="sa"))
    b = run_view(make_summary("b", n=30, sha="sb"))
    a.case_ids, b.case_ids = frozenset({"A"}), frozenset({"B"})
    a.systems["agentic"].lat_avg_ms, b.systems["agentic"].lat_avg_ms = 1000.0, 3000.0
    assert pool([a, b]).systems["agentic"].lat_avg_ms == pytest.approx(2500.0)


def test_pool_refuses_runs_whose_settings_differ():
    a, b = _disjoint("a", "b")
    b.settings = {**b.settings, "rerank_enabled": True}
    with pytest.raises(ValueError, match=r"settings differ \(rerank_enabled\)"):
        pool([a, b])


def test_pool_ignores_settings_that_only_shape_the_intervals():
    a, b = _disjoint("a", "b")
    b.settings = {**b.settings, "eval_bootstrap_seed": 99, "eval_confidence": 0.9}
    assert pool([a, b]).n_questions == 80


def test_pool_refuses_the_same_dataset_and_split_twice():
    a = view(run_id="a", sha="same", split="test")
    b = view(run_id="b", sha="same", split="test")
    with pytest.raises(ValueError, match="counted twice"):
        pool([a, b])


def test_pool_refuses_overlapping_questions_even_when_the_hash_differs():
    a, b = _disjoint("a", "b")
    b.case_ids = frozenset({"F02", "D09"})
    with pytest.raises(ValueError, match="both ran question.*F02.*counted twice"):
        pool([a, b])


def test_pool_refuses_different_systems_and_a_single_run():
    a, b = _disjoint("a", "b")
    b2 = run_view(make_summary("b2", sha="sb2", extra_mode=True))
    b2.case_ids = frozenset({"X"})
    with pytest.raises(ValueError, match="different systems"):
        pool([a, b2])
    with pytest.raises(ValueError, match="at least two"):
        pool([a])


# ------------------------------------------------------------------------------ loading
def _write_run(root: Path, run_id: str, cases: list[str] | None = None, *, raw_cases: str | None = None) -> Path:
    folder = root / run_id
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(make_summary(run_id)), encoding="utf-8")
    if raw_cases is not None:
        (folder / "cases.jsonl").write_text(raw_cases, encoding="utf-8")
    elif cases is not None:
        (folder / "cases.jsonl").write_text("\n".join(json.dumps({"case_id": c, "mode": m}) for c in cases for m in ("baseline", "agentic")) + "\n", encoding="utf-8")
    return folder


def test_run_is_found_by_id_folder_or_file_and_ids_are_read(tmp_path):
    folder = _write_run(tmp_path, "fresh-1", ["F01", "F02"])
    for ref in ("fresh-1", folder, folder / "summary.json"):
        v = load_run_view(ref, tmp_path)
        assert v.run_id == "fresh-1" and v.case_ids == frozenset({"F01", "F02"})
    assert load_summary("fresh-1", tmp_path)["run_id"] == "fresh-1"


def test_missing_run_and_unreadable_cases_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no summary.json for run 'nope'"):
        load_run_view("nope", tmp_path)
    no_cases = _write_run(tmp_path, "a")
    assert read_case_ids(no_cases / "summary.json") is None
    broken = _write_run(tmp_path, "b", raw_cases='{"case_id": "F01"}\nnot json\n')
    assert read_case_ids(broken / "summary.json") is None


# ------------------------------------------------------------------------------ command-line script
def _script():
    spec = importlib.util.spec_from_file_location("render_results_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_prints_the_table_and_writes_the_file(tmp_path, capsys):
    _write_run(tmp_path, "fresh-1", ["F01"])
    out = tmp_path / "tables.md"
    assert _script().main(["fresh-1", "--results-dir", str(tmp_path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| Overall correct | 75% (30/40) | 90% (36/40) |" in printed
    assert out.read_text(encoding="utf-8").startswith("### Run `fresh-1`")


def test_script_exits_2_with_a_reason_for_unfinished_runs_and_bad_pools(tmp_path, capsys):
    folder = _write_run(tmp_path, "half", ["F01"])
    data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    data["status"], data["progress"]["units_recorded"] = "stopped", 10
    (folder / "summary.json").write_text(json.dumps(data), encoding="utf-8")
    assert _script().main(["half", "--results-dir", str(tmp_path)]) == 2
    assert "not complete" in capsys.readouterr().err
    _write_run(tmp_path, "full", ["F01"])
    assert _script().main(["full", "full", "--pool", "--results-dir", str(tmp_path)]) == 2
    assert "counted twice" in capsys.readouterr().err
    assert _script().main(["missing", "--results-dir", str(tmp_path)]) == 2


# ------------------------------------------------------------------------------ the row without the conflicting category (finding 10)
CONFLICTING_ROW = "Overall correct, without the conflicting category"


def _write_scored_run(root: Path, run_id: str, outcomes: dict[str, dict[str, tuple[str, bool]]], *, den: int | None = None) -> Path:
    """A run folder whose summary totals agree with ``outcomes`` (``{system: {case_id: (category, correct)}}``)."""
    folder = root / run_id
    folder.mkdir(parents=True)
    n = den if den is not None else len(outcomes["baseline"])
    totals = {m: (sum(ok for _, ok in rows.values()), n) for m, rows in outcomes.items()}
    (folder / "summary.json").write_text(json.dumps(make_summary(run_id, n=n, overall=(totals["baseline"], totals["agentic"]), sha=f"sha-{run_id}")), encoding="utf-8")
    lines = [json.dumps({"case_id": c, "category": cat, "mode": m, "scores": {"correct": ok}}) for m, rows in outcomes.items() for c, (cat, ok) in rows.items()]
    (folder / "cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def _with_conflicting() -> dict[str, dict[str, tuple[str, bool]]]:
    """Three conflicting cases plus seven others: baseline misses all three and one other (6/10), agentic gets one conflicting and every other (8/10)."""
    others = [f"D{i:02d}" for i in range(1, 8)]
    c = CONFLICTING_CATEGORY
    return {
        "baseline": {"C01": (c, False), "C02": (c, False), "C03": (c, False), **{o: ("direct_kb", o != "D07") for o in others}},
        "agentic": {"C01": (c, False), "C02": (c, True), "C03": (c, False), **{o: ("direct_kb", True) for o in others}},
    }


def test_the_excluded_category_is_the_conflicting_one_named_in_finding_10():
    assert CONFLICTING_CATEGORY == "conflicting" and WITHOUT_CONFLICTING == ("answer", "overall_correct_without_conflicting")


def test_row_without_the_conflicting_category_is_shown_beside_the_overall_rate(tmp_path):
    _write_scored_run(tmp_path, "r", _with_conflicting())
    text = render_run(load_run_view("r", tmp_path))
    assert "| Overall correct | 60% (6/10) | 80% (8/10) |" in text
    assert f"| {CONFLICTING_ROW} | 86% (6/7) | 100% (7/7) |" in text                  # baseline 6 correct of 7, agentic 7 of 7
    assert f"| {CONFLICTING_ROW} | 86% [" in text                                         # and the interval table carries it too
    assert "C01-C06, F29-F31" in text and "No claim about conflicting documents" in text


def test_the_category_decides_not_the_case_id(tmp_path):
    rows = {m: {f"D{i:02d}": ("direct_kb", True) for i in range(1, 9)} for m in ("baseline", "agentic")}
    for m in rows:
        rows[m]["F29"] = (CONFLICTING_CATEGORY, False)                                  # a fresh-block conflicting case is excluded
        rows[m]["C99"] = ("direct_kb", False)                                           # an id that looks like one is not, unless its category says so
    _write_scored_run(tmp_path, "r", rows)
    text = render_run(load_run_view("r", tmp_path))
    assert f"| {CONFLICTING_ROW} | 89% (8/9) | 89% (8/9) |" in text


def test_row_is_left_out_when_nothing_was_excluded(tmp_path):
    ids = {m: {f"D{i:02d}": ("direct_kb", i % 2 == 0) for i in range(1, 11)} for m in ("baseline", "agentic")}
    _write_scored_run(tmp_path, "r", ids)
    text = render_run(load_run_view("r", tmp_path))
    assert CONFLICTING_ROW not in text and "F29-F31" not in text


def test_row_needs_case_outcomes_and_every_unit_scored(tmp_path):
    _write_run(tmp_path, "no-cases")                                                # summary only: nothing to derive from
    assert CONFLICTING_ROW not in render_run(load_run_view("no-cases", tmp_path))
    partial = _with_conflicting()
    partial["baseline"].pop("D07")                                                  # one baseline unit unscored: 9 rows for 10 questions
    _write_scored_run(tmp_path, "partial", partial, den=10)
    text = render_run(load_run_view("partial", tmp_path))
    assert f"| {CONFLICTING_ROW} | n/a | 100% (7/7) |" in text                       # the baseline cannot be split, the agentic system can


def test_pooling_adds_the_derived_counts_and_needs_every_run_to_have_them(tmp_path):
    _write_scored_run(tmp_path, "a", _with_conflicting())
    other = {m: {f"D{i:02d}": ("direct_kb", True) for i in range(11, 21)} for m in ("baseline", "agentic")}
    other["baseline"]["D20"] = ("direct_kb", False)                                  # baseline 9/10, agentic 10/10, no conflicting cases
    _write_scored_run(tmp_path, "b", other)
    a, b = load_run_view("a", tmp_path), load_run_view("b", tmp_path)
    text = render_run(pool([a, b]))
    assert f"| {CONFLICTING_ROW} | 88% (15/17) | 100% (17/17) |" in text                # (6+9)/(7+10) and (7+10)/(7+10)
    plain = run_view(make_summary("c", sha="sha-c"))
    plain.case_ids = frozenset({"Z01"})                                              # no derived rate: its questions cannot be split
    assert CONFLICTING_ROW not in render_run(pool([a, plain]))
