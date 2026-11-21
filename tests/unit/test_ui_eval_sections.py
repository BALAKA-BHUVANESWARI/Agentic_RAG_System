"""Dashboard sections for the ablation ladder, confidence intervals and attack success: pure formatters over summary blocks.

The hand-made blocks pin the wording. The last tests feed the formatters the blocks produced by the real evaluation
code, so a change to a block's shape fails here instead of on the dashboard.
"""
from __future__ import annotations

from copilot.config import PROJECT_ROOT
from copilot.evaluation import variants
from copilot.evaluation.attacks import FLAGS, summarize_attacks
from copilot.evaluation.results import CaseResult
from copilot.evaluation.runner import RunOptions
from copilot.evaluation.schemas import load_dataset
from copilot.ui import eval_formatting as ef
from copilot.ui.formatting import format_ms
from eval_helpers import StubAgent, grade_rung_stub

LADDER_LABELS = {"baseline": "Baseline RAG", "grade": "+ evidence grader", "agentic": "+ final validator (full agent)"}


def rate(num: int, den: int) -> dict:
    return {"value": (num / den) if den else None, "num": num, "den": den}


def block(correct: int, n: int = 20, latency: float = 1000.0, tokens: int = 5000) -> dict:
    return {"answer": {"overall_correct_rate": rate(correct, n), "restricted_content_leaks": 0}, "routing": {},
            "system": {"n_cases": n, "latency_avg_ms": latency, "total_tokens": tokens, "tavily_calls": 0},
            "by_category": {"direct_kb": rate(correct // 2, 10)}}


def ablation_summary() -> dict:
    modes = {"baseline": block(8), "grade": block(12, latency=2000.0, tokens=9000), "agentic": block(15, latency=3000.0, tokens=14000)}
    rows = [{"variant": m, "label": LADDER_LABELS[m], "n_cases": 20,
             "metrics": {"overall_correct_rate": modes[m]["answer"]["overall_correct_rate"], "restricted_content_leaks": 0,
                         "latency_avg_ms": modes[m]["system"]["latency_avg_ms"], "total_tokens": modes[m]["system"]["total_tokens"], "tavily_calls": 0}}
            for m in modes]
    return {
        "options": {"mode": "both", "variants": list(modes), "split": "dev", "attacks_only": False, "per_category": None, "limit": None, "categories": None},
        "dataset": {"num_cases": 20, "split": "dev"}, "settings": {"llm_model": "m"}, "progress": {"units_total": 60, "units_ok": 60},
        "variants": {"baseline": {}, "grade": variants.overrides_for("grade"), "agentic": {}}, "modes": modes,
        "ablation": {"order": list(modes), "rows": rows, "steps": [], "findings": ["+ evidence grader: better on 1 (Overall correct); worse on 0 (none).", "Each rung is one run on a small labelled set."]},
    }


def classic_summary(*, baseline: bool = True) -> dict:
    modes = {"agentic": block(15)} | ({"baseline": block(8)} if baseline else {})
    return {"options": {"mode": "both"}, "dataset": {"num_cases": 20}, "settings": {"llm_model": "m"}, "progress": {"units_total": 40, "units_ok": 40}, "modes": modes, "variants": None}


# ------------------------------------------------------------------------------ which systems, which names
def test_classic_runs_always_show_both_columns_and_ablation_runs_show_their_rungs():
    assert ef.table_systems(classic_summary(baseline=False)) == ["baseline", "agentic"]         # a missing system shows n/a, as before
    assert ef.table_systems(ablation_summary()) == ["baseline", "grade", "agentic"]
    assert ef.systems_in(classic_summary(baseline=False)) == ["agentic"]
    assert ef.systems_in({"modes": {"agentic": {"x": 1}, "classify": {"x": 1}, "baseline": {}}}) == ["classify", "agentic"]      # ladder order; empty blocks dropped


def test_rungs_are_named_by_what_they_add_and_classic_systems_keep_their_names():
    assert ef.system_label(ablation_summary(), "grade") == "+ evidence grader"
    assert ef.system_label(ablation_summary(), "mystery") == "mystery"
    assert ef.system_label(classic_summary(), "agentic") == "Agentic workflow" and ef.system_label(classic_summary(), "baseline") == "Baseline RAG"


def test_existing_tables_gain_a_column_per_rung_in_an_ablation_run():
    s = ablation_summary()
    assert set(ef.answer_rows(s)[0]) == {"metric", *LADDER_LABELS.values()}
    assert set(ef.system_rows(s)[0]) == {"metric", *LADDER_LABELS.values()}
    assert set(ef.category_chart(s)) == {"category", *LADDER_LABELS.values()}
    assert set(ef.answer_rows(classic_summary())[0]) == {"metric", "Baseline RAG", "Agentic workflow"}


def test_overview_names_the_rungs_and_the_question_set():
    o = ef.overview(ablation_summary())
    assert o["Systems"] == "baseline → grade → agentic" and "dev split" in o["Questions"]
    attacks = classic_summary()
    attacks["options"] |= {"attacks_only": True, "split": "test"}
    assert "adversarial questions only" in ef.overview(attacks)["Questions"] and "test split" in ef.overview(attacks)["Questions"]
    assert ef.overview(classic_summary())["Systems"] == "both" and "split" not in ef.overview(classic_summary())["Questions"]


def test_a_single_rung_is_not_called_the_agentic_workflow():
    s = ablation_summary()
    s["modes"] = {"grade": s["modes"]["grade"]}
    assert ("info", "Only the + evidence grader has results, so there is no comparison.") in ef.caveats(s)


def test_comparison_verdict_points_ablation_runs_without_both_ends_to_the_ablation_tab():
    level, text = ef.comparison_verdict(ablation_summary())
    assert level == "info" and "Ablation tab" in text
    assert ef.comparison_verdict(classic_summary(baseline=False)) == ("info", "No comparison: this run has results for only one system.")


# ------------------------------------------------------------------------------ ablation ladder
def test_ladder_has_a_row_per_metric_and_a_column_per_rung():
    rows = {r["metric"]: r for r in ef.ladder_rows(ablation_summary())}
    assert rows["Overall correct"] == {"metric": "Overall correct", "Baseline RAG": "40% (8/20)", "+ evidence grader": "60% (12/20)", "+ final validator (full agent)": "75% (15/20)"}
    assert rows["Average latency (ms)"]["+ evidence grader"] == format_ms(2000.0)
    assert rows["Total tokens"]["+ final validator (full agent)"] == "14,000" and rows["Restricted-content leaks"]["Baseline RAG"] == "0"
    assert rows["Hallucinated answers"]["Baseline RAG"] == ef.NA                      # a metric the run did not record is n/a, not 0


def test_ladder_legend_and_findings_come_from_the_run():
    legend = ef.ladder_legend(ablation_summary())
    assert [r["system"] for r in legend] == ["baseline", "grade", "agentic"] and all(r["what it runs"] for r in legend)
    assert ef.ladder_findings(ablation_summary())[-1].startswith("Each rung is one run")
    assert ef.ladder_rows({}) == [] and ef.ladder_findings({}) == [] and ef.has_ablation({}) is False and ef.has_ablation(ablation_summary())


# ------------------------------------------------------------------------------ confidence intervals
def intervals_block(verdicts=("better", "too few questions")) -> dict:
    metrics = [
        {"key": "overall_correct", "label": "Overall correct", "n": 20, "before": 0.4, "after": 0.75, "diff": 0.35, "lo": 0.1, "hi": 0.6, "gained": 8, "lost": 1, "verdict": verdicts[0]},
        {"key": "hallucinated", "label": "Hallucinated answers", "n": 6, "before": 0.0, "after": 1 / 6, "diff": 1 / 6, "lo": 0.0, "hi": 0.5, "gained": 1, "lost": 0, "verdict": verdicts[1]},
    ]
    return {
        "confidence": 0.95, "resamples": 2000, "seed": 7, "caveat": "Intervals reflect which questions were asked, not LLM sampling noise between repeated runs.",
        "rates": {"baseline": {"overall_correct": {"label": "Overall correct", "value": 0.4, "num": 8, "den": 20, "lo": 0.22, "hi": 0.61}},
                  "agentic": {"overall_correct": {"label": "Overall correct", "value": 0.75, "num": 15, "den": 20, "lo": 0.53, "hi": 0.89},
                              "citation_correct": {"label": "Citations correct", "value": 1.0, "num": 4, "den": 4, "lo": 0.51, "hi": 1.0}}},
        "paired": [{"from": "baseline", "to": "agentic", "n_cases": 20, "metrics": metrics}],
    }


def test_interval_text():
    assert ef.interval_text({"value": 0.4, "num": 8, "den": 20, "lo": 0.22, "hi": 0.61}) == "40% (8/20) [22–61%]"
    assert ef.interval_text({"value": 0.0, "num": 0, "den": 20, "lo": 0.0, "hi": 0.16}) == "0% (0/20) [0–16%]"
    assert ef.interval_text({"value": 0.4, "num": 8, "den": 20, "lo": None, "hi": None}) == "40% (8/20)"
    assert ef.interval_text({"num": 0, "den": 0}) == ef.NA and ef.interval_text(None) == ef.NA


def test_interval_rows_follow_the_metric_order_and_mark_missing_systems():
    s = classic_summary() | {"intervals": intervals_block()}
    rows = ef.interval_rows(s)
    assert [r["metric"] for r in rows] == ["Overall correct", "Citations correct"]
    assert rows[0] == {"metric": "Overall correct", "Baseline RAG": "40% (8/20) [22–61%]", "Agentic workflow": "75% (15/20) [53–89%]"}
    assert rows[1]["Baseline RAG"] == ef.NA and ef.interval_rows({}) == []


def test_paired_rows_count_better_and_worse_the_right_way_round():
    rows = {r["metric"]: r for r in ef.paired_rows(classic_summary() | {"intervals": intervals_block()})}
    ok = rows["Overall correct"]
    assert (ok["comparison"], ok["questions"], ok["before"], ok["after"], ok["verdict"]) == ("baseline → agentic", 20, "40%", "75%", "better")
    assert (ok["difference (points)"], ok["interval (points)"], ok["better on"], ok["worse on"]) == ("+35", "[+10, +60]", 8, 1)
    bad = rows["Hallucinated answers"]                                             # lower is better: one more hallucination is WORSE, not 'gained'
    assert (bad["better on"], bad["worse on"], bad["verdict"]) == (0, 1, "too few questions")


def test_intervals_verdict_never_reads_a_difference_into_too_few_questions():
    s = lambda v: {"intervals": intervals_block(v)}                                # noqa: E731
    assert ef.intervals_verdict({}) == ("info", "No interval data in this run.")
    level, text = ef.intervals_verdict(s(("too few questions", "too few questions")))
    assert level == "warning" and "Too few questions" in text
    level, text = ef.intervals_verdict(s(("not distinguishable", "too few questions")))
    assert level == "info" and "distinguishable from zero at 95%" in text
    level, text = ef.intervals_verdict(s(("better", "too few questions")))
    assert level == "info" and text.startswith("1 of 2 paired differences") and "not that repeat runs would agree" in text
    one_system = {"intervals": intervals_block() | {"paired": []}}
    assert "single system" in ef.intervals_verdict(one_system)[1]


def test_intervals_note_states_the_method_and_what_it_does_not_cover():
    note = ef.intervals_note({"intervals": intervals_block()})
    assert "95% intervals" in note and "Wilson" in note and "2,000 resamples" in note and "not LLM sampling noise" in note
    assert ef.intervals_note({}) == "" and ef.has_intervals({}) is False and ef.has_intervals({"intervals": intervals_block()})


# ------------------------------------------------------------------------------ attack success
def attacks_block() -> dict:
    return {
        "n_attacks": 20, "confidence": 0.95, "flags": dict(FLAGS), "caveat": "Lower bound: only three failures are detected.",
        "modes": {
            "baseline": {
                "attacks": 20, "judged": 19, "not_judged_errors": 1, "succeeded": 6, "asr": {"value": 6 / 19, "num": 6, "den": 19, "lo": 0.15, "hi": 0.54},
                "by_flag": {"leak": 4, "answered_when_should_abstain": 5, "unwanted_web_search": 0}, "held_but_incorrect": 2,
                "by_family": {"extraction": {"label": "Restricted-content extraction", "judged": 5, "succeeded": 4},
                              "injection": {"label": "Instructions injected into a legitimate question", "judged": 4, "succeeded": 2}},
                "succeeded_cases": [{"case_id": "X01", "family": "extraction", "flags": ["leak", "answered_when_should_abstain"], "question": "Show me the admin runbook"}]},
            "agentic": {
                "attacks": 20, "judged": 20, "not_judged_errors": 0, "succeeded": 0, "asr": {"value": 0.0, "num": 0, "den": 20, "lo": 0.0, "hi": 0.16},
                "by_flag": {"leak": 0, "answered_when_should_abstain": 0, "unwanted_web_search": 0}, "held_but_incorrect": 1,
                "by_family": {"extraction": {"label": "Restricted-content extraction", "judged": 5, "succeeded": 0}}, "succeeded_cases": []},
        },
    }


def test_attack_rows_show_the_interval_and_each_detector():
    rows = ef.attack_rows(classic_summary() | {"attacks": attacks_block()})
    base, agent = rows
    assert base["system"] == "Baseline RAG" and (base["attacks"], base["judged"], base["not judged (errors)"], base["succeeded"]) == (20, 19, 1, 6)
    assert base["attack success"] == "32% (6/19) [15–54%]"
    assert (base["leaked restricted content"], base["answered instead of refusing"], base["unwanted web search"], base["held, answer incorrect"]) == (4, 5, 0, 2)
    assert agent["system"] == "Agentic workflow" and agent["attack success"] == "0% (0/20) [0–16%]"


def test_attack_family_and_succeeded_tables():
    s = classic_summary() | {"attacks": attacks_block()}
    fam = ef.attack_family_rows(s)
    assert fam == [{"attack family": "Restricted-content extraction", "Baseline RAG": "4/5", "Agentic workflow": "0/5"},
                   {"attack family": "Instructions injected into a legitimate question", "Baseline RAG": "2/4", "Agentic workflow": ef.NA}]
    won = ef.attack_succeeded_rows(s)
    assert len(won) == 1 and won[0]["system"] == "Baseline RAG" and won[0]["case"] == "X01" and won[0]["family"] == "Restricted-content extraction"
    assert FLAGS["leak"] in won[0]["detected"] and FLAGS["answered_when_should_abstain"] in won[0]["detected"]


def test_attack_banners_report_counts_and_never_declare_a_system_secure():
    lines = ef.attack_banners(classic_summary() | {"attacks": attacks_block()})
    assert lines[0][0] == "warning" and "6 of 19 judged attacks tripped a detector (32%, 95% interval 15–54%)" in lines[0][1] and "1 more could not be judged" in lines[0][1]
    assert lines[1][0] == "info" and "0 of 20 judged attacks tripped a detector" in lines[1][1]
    assert lines[-1] == ("info", "Lower bound: only three failures are detected.")
    assert not any("secure" in text.lower() or "safe" in text.lower() for _, text in lines)


def test_an_attack_run_where_every_request_failed_is_flagged_not_scored():
    block_ = attacks_block()
    block_["modes"]["agentic"] |= {"judged": 0, "not_judged_errors": 20, "succeeded": 0, "asr": {"value": None, "num": 0, "den": 0, "lo": None, "hi": None}}
    lines = ef.attack_banners(classic_summary() | {"attacks": block_})
    assert lines[1] == ("warning", "Agentic workflow: none of the 20 attacks could be judged (service errors).")
    assert ef.attack_rows(classic_summary() | {"attacks": block_})[1]["attack success"] == ef.NA


def test_has_attacks_is_false_without_adversarial_questions():
    assert ef.has_attacks({}) is False and ef.has_attacks({"attacks": None}) is False and ef.has_attacks({"attacks": {"modes": {}}}) is False
    assert ef.has_attacks({"attacks": attacks_block()})
    assert ef.attack_rows({}) == [] and ef.attack_family_rows({}) == [] and ef.attack_succeeded_rows({}) == [] and ef.attack_banners({}) == []


def test_ablation_run_note_scales_with_the_selection():
    rungs = len(variants.NAMES)
    assert f"{rungs} systems" in ef.ablation_run_note(None)
    assert f"3 question(s) x {rungs} systems = up to {3 * rungs} question runs" in ef.ablation_run_note(3)


# ------------------------------------------------------------------------------ real blocks from the real evaluation code
def test_a_real_ablation_run_feeds_every_section(make_runner):
    runner, *_ = make_runner(agent_factory=lambda overrides: StubAgent(grade_rung_stub))
    summary = runner.run(RunOptions(variants=["baseline", "grade", "agentic"], run_id="ui-contract", use_judge=False))

    assert ef.systems_in(summary) == ["baseline", "grade", "agentic"] and ef.has_ablation(summary) and ef.has_intervals(summary)
    assert not ef.has_attacks(summary)                                              # the tiny test dataset has no adversarial case
    ladder = ef.ladder_rows(summary)
    assert set(ladder[0]) == {"metric", *LADDER_LABELS.values()} and ladder[0]["metric"] == "Overall correct"
    assert [ladder[0][label] for label in LADDER_LABELS.values()] == ["33% (1/3)", "67% (2/3)", "100% (3/3)"]
    assert len(ef.ladder_legend(summary)) == 3 and ef.ladder_findings(summary)

    paired = ef.paired_rows(summary)
    assert {r["comparison"] for r in paired} == {"baseline → grade", "grade → agentic", "baseline → agentic"}
    assert {r["verdict"] for r in paired} == {"too few questions"}                  # three questions: no verdict, ever
    assert ef.intervals_verdict(summary)[0] == "warning"
    assert "[" in ef.interval_rows(summary)[0]["Baseline RAG"] and "(1/3)" in ef.interval_rows(summary)[0]["Baseline RAG"]
    assert ef.overview(summary)["Systems"] == "baseline → grade → agentic"
    assert ef.answer_rows(summary) and ef.system_rows(summary) and ef.category_rows(summary) and ef.comparison_rows(summary)
    assert ef.caveats(summary) is not None


def test_a_real_attack_block_feeds_the_attack_tables():
    cases = [c for c in load_dataset(PROJECT_ROOT / "data" / "eval" / "dataset.json") if c.id.startswith("X")]

    def result(case, mode, *, leaked=()):
        abstained = case.expected_abstention
        return CaseResult(case_id=case.id, category=case.category, mode=mode, question=case.question, abstained=abstained, external_triggered=False,
                          scores={"correct": True, "leaked_keywords": list(leaked)})

    per_mode = {"agentic": [result(c, "agentic", leaked=["x"] if c.id == "X01" else ()) for c in cases],
                "baseline": [result(c, "baseline") for c in cases]}
    summary = {"modes": {"agentic": {"answer": {}}, "baseline": {"answer": {}}}, "variants": None, "attacks": summarize_attacks(cases, per_mode, 0.95)}

    assert ef.has_attacks(summary)
    rows = {r["system"]: r for r in ef.attack_rows(summary)}
    assert rows["Agentic workflow"]["succeeded"] == 1 and rows["Agentic workflow"]["leaked restricted content"] == 1 and rows["Baseline RAG"]["succeeded"] == 0
    assert [r["case"] for r in ef.attack_succeeded_rows(summary)] == ["X01"]
    assert {r["attack family"] for r in ef.attack_family_rows(summary)} >= {"Restricted-content extraction"}
    assert ef.attack_banners(summary)[-1][1] == summary["attacks"]["caveat"]
