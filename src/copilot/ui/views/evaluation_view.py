"""Evaluation dashboard (Admin > Evaluation): start and watch runs, then read their results.

The page only talks to the API. Every number comes from the run's own ``summary.json`` through
``copilot.ui.eval_formatting``; nothing is hard-coded, and the page says plainly when a run is too
small, partial or on a different model to support a conclusion.
"""
from __future__ import annotations

import json
import time

import streamlit as st

from copilot.config import Settings
from copilot.evaluation import variants as variant_registry
from copilot.evaluation.schemas import CATEGORIES
from copilot.ui import eval_formatting as ef
from copilot.ui.api_client import ApiError, CopilotClient

_LEVELS = {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}
ABLATION_CHOICE = "ablation ladder"


def _banner(level: str, text: str) -> None:
    _LEVELS.get(level, st.info)(text)


def _api_error(exc: ApiError) -> None:
    (st.info if exc.status_code == 501 else st.error)(exc.detail)


def _summary_for(client: CopilotClient, run: dict) -> dict | None:
    """The run's summary. Finished runs are cached in the session; a running run is read fresh each time."""
    cache: dict[str, dict] = st.session_state.setdefault("eval_summaries", {})
    key = f"{run['run_id']}|{run['status']}|{run.get('finished_at')}"  # a run resumed elsewhere changes status/finish time, so it is re-read
    if key in cache:
        return cache[key]
    try:
        summary = client.evaluation_summary(run["run_id"])
    except ApiError as exc:
        if exc.status_code != 404:
            _api_error(exc)
        return None
    if run["status"] not in ef.ACTIVE:
        cache[key] = summary
    return summary


# ------------------------------------------------------------------------------ run controls
def _controls(client: CopilotClient, runs: list[dict], active: dict | None) -> None:
    st.markdown("#### Start a run")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox(
        "Systems", ["both", "agentic", "baseline", ABLATION_CHOICE], key="eval_mode",
        help="'both' runs the baseline RAG and the agentic workflow on the same questions. 'ablation ladder' runs six systems, each adding one agent step to the one before.",
    )
    per_cat = c2.number_input("Questions per category (0 = all)", min_value=0, max_value=100, value=1, step=1, key="eval_per_category")
    limit = c3.number_input("Max questions (0 = no limit)", min_value=0, max_value=1000, value=0, step=1, key="eval_limit")
    categories = st.multiselect("Categories", list(CATEGORIES), default=list(CATEGORIES), key="eval_categories")
    s1, s2 = st.columns(2)
    split = s1.selectbox("Question set", ["all", "dev", "test"], key="eval_split", help="Frozen dev/test split of the labelled questions (docs/EVAL_SPLIT.md): tune on dev, report test.")
    attacks_only = s2.checkbox("Adversarial questions only (X01-X20)", value=False, key="eval_attacks_only", help="Runs only the attack questions and adds an attack-success tab.")
    use_judge = st.checkbox("Also run the LLM judge (faithfulness and relevance; uses extra tokens)", value=False, key="eval_judge")

    per_cat_v, limit_v = per_cat or None, limit or None
    cats_v = categories if categories and set(categories) != set(CATEGORIES) else None
    ref = ef.latest_completed(runs)
    ref_summary = _summary_for(client, ref) if ref else None
    ablation = mode == ABLATION_CHOICE
    n_selected = ef.selected_question_count(per_cat_v, limit_v, categories or None)
    estimate = None if ablation else ef.estimate_tokens(ref_summary, n_selected, mode, use_judge)
    if ablation:
        st.caption(ef.ablation_run_note(n_selected))
    elif estimate:
        st.caption(f"Rough token need: up to about {estimate:,}, scaled from run {ref['run_id']}. Free provider tiers have daily token limits.")
    else:
        st.caption("Real calls to Groq, Pinecone and Tavily. A full run can use a large share of a free daily token limit; try a small run first.")

    if st.button("Start evaluation", key="eval_start", disabled=active is not None or not categories):
        try:
            info = client.start_evaluation(
                mode="both" if ablation else mode, variants=list(variant_registry.NAMES) if ablation else None, limit=limit_v,
                per_category=per_cat_v, categories=cats_v, use_judge=use_judge, split=split, attacks_only=attacks_only,
            )
        except ApiError as exc:
            (st.warning if exc.status_code == 409 else st.error)(exc.detail)
        else:
            st.session_state["eval_pick"] = info["run_id"]
            st.rerun()

    resumable = ef.resumable_runs(runs)
    if resumable:
        r1, r2 = st.columns([3, 1])
        chosen = r1.selectbox("Resume an unfinished run", resumable, format_func=ef.run_label, key="eval_resume_pick")
        if r2.button("Resume", key="eval_resume", disabled=active is not None):
            try:
                client.start_evaluation(resume_run_id=chosen["run_id"])
            except ApiError as exc:
                (st.warning if exc.status_code == 409 else st.error)(exc.detail)
            else:
                st.session_state.pop("eval_summaries", None)  # the run will rewrite its summary
                st.session_state["eval_pick"] = chosen["run_id"]
                st.rerun()


def _active_panel(client: CopilotClient, active: dict) -> None:
    st.markdown("#### Run in progress")
    st.write(f"`{active['run_id']}`")
    st.progress(ef.progress_fraction(active), text=f"{active.get('done', 0)}/{active.get('total', 0)} · {active.get('message') or ''}")
    c1, c2 = st.columns([1, 3])
    if c1.button("Cancel run", key="eval_cancel", disabled=active["status"] == "cancelling"):
        try:
            client.cancel_evaluation(active["run_id"])
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)
    c2.checkbox("Pause automatic refresh", key="eval_pause_refresh", help="While a run is active the page refreshes by itself; tick this to stop that.")


# ------------------------------------------------------------------------------ results
def _table(rows: list[dict]) -> None:
    st.dataframe(rows, width="stretch", hide_index=True)


def _overview_tab(run: dict, summary: dict) -> None:
    cols = st.columns(3)
    for i, (label, value) in enumerate(ef.overview(summary).items()):
        cols[i % 3].metric(label, value)
    for level, text in ef.caveats(summary):
        _banner(level, text)
    for note in summary.get("notes") or []:
        st.caption(note)
    st.download_button("Download summary.json", json.dumps(summary, indent=2), file_name=f"{run['run_id']}_summary.json", mime="application/json", key="eval_download")


def _retrieval_tab(summary: dict) -> None:
    n = (summary.get("retrieval") or {}).get("n_cases", 0)
    if not n:
        st.info("No retrieval metrics: no question in this run names the expected documents.")
        return
    st.caption(f"{n} question(s) with labelled expected documents. Chunk recall is measured against the ranked pool, not every chunk in the index.")
    c = st.columns(2)
    for col, (label, value) in zip(c, ef.retrieval_mrr(summary).items()):
        col.metric(label, value)
    _table(ef.retrieval_rows(summary))


def _answer_tab(summary: dict) -> None:
    st.caption("Deterministic checks (answer points, forbidden-content leaks, citations) plus the optional LLM judge. Correct abstention is scored on questions the system should not answer.")
    _table(ef.answer_rows(summary))


def _routing_tab(summary: dict) -> None:
    if not (summary.get("modes") or {}).get("agentic"):
        st.info("Routing metrics exist only for the agentic workflow; this run has no agentic results.")
        return
    _table(ef.routing_rows(summary))
    st.markdown("**External search and rewriting**")
    _table(ef.external_stats(summary))
    cal = ef.calibration_rows(summary)
    if cal:
        st.markdown("**Grader confidence versus accuracy**")
        st.caption("The grader's confidence is a decision signal, not a calibrated probability. Compare it with how often the grade was right.")
        _table(cal)


def _system_tab(summary: dict) -> None:
    st.caption("Cost is approximate. Latency includes provider rate-limit waits and retries.")
    _table(ef.system_rows(summary))
    ing = summary.get("ingestion") or {}
    if ing:
        st.markdown("**Ingestion**")
        c = st.columns(2)
        c[0].metric("Documents currently indexed", ef.rate_text(ing.get("documents")))
        c[1].metric("Ingestion attempts that succeeded", ef.rate_text(ing.get("attempts")), help="Includes historical failed attempts, so it can be lower than the document rate.")


def _comparison_tab(summary: dict) -> None:
    level, text = ef.comparison_verdict(summary)
    _banner(level, text)
    rows = ef.comparison_rows(summary)
    if not rows:
        return
    for line in ef.comparison_findings(summary):
        st.write(f"- {line}")
    _table(rows)
    st.markdown("**Overall correct by question category**")
    chart = ef.category_chart(summary)
    if chart["category"]:
        import pandas as pd  # a Streamlit dependency

        st.bar_chart(pd.DataFrame(chart).set_index("category"))
        _table(ef.category_rows(summary))
        st.caption("Each category has only a few questions, so single-category differences are very noisy.")


def _ablation_tab(summary: dict) -> None:
    st.caption("Each system adds one agent step to the one before it. The routing rules (private first, web search only when allowed) are the same in every rung.")
    _table(ef.ladder_rows(summary))
    st.markdown("**What each system runs**")
    _table(ef.ladder_legend(summary))
    st.markdown("**Reading the steps**")
    for line in ef.ladder_findings(summary):
        st.write(f"- {line}")
    if ef.has_intervals(summary):
        st.caption("Whether a step is distinguishable from noise is on the Confidence intervals tab.")


def _intervals_tab(summary: dict) -> None:
    _banner(*ef.intervals_verdict(summary))
    st.caption(ef.intervals_note(summary))
    st.markdown("**Each system's rates with an interval**")
    _table(ef.interval_rows(summary))
    paired = ef.paired_rows(summary)
    if paired:
        st.markdown("**Paired differences between systems (same questions)**")
        st.caption("'better on' and 'worse on' count questions where the later system did better or worse than the earlier one. A verdict is given only when the interval excludes zero and at least 10 questions contribute.")
        _table(paired)


def _attacks_tab(summary: dict) -> None:
    for level, text in ef.attack_banners(summary):
        _banner(level, text)
    _table(ef.attack_rows(summary))
    st.markdown("**By attack family** (succeeded / judged)")
    _table(ef.attack_family_rows(summary))
    won = ef.attack_succeeded_rows(summary)
    if won:
        st.markdown("**Attacks that tripped a detector**")
        _table(won)
    else:
        st.success("No attack tripped a detector in this run.")


def _failed_tab(summary: dict) -> None:
    cases = summary.get("failed_cases") or []
    if not cases:
        st.success("No failed or incorrect cases.")
        return
    st.caption(f"{len(cases)} failed or incorrect case(s). Expand one to see the answer the system gave and the path it took.")
    _table(ef.failed_rows(summary))
    for case in cases:
        with st.expander(ef.failed_case_title(case)):
            st.write(case["question"])
            st.caption(f"Expected route: {case.get('expected_route')} · reason: {case.get('abstention_reason') or case.get('error') or '-'} · answer points {case.get('points_met', 0)}/{case.get('points_total', 0)}")
            st.markdown("**Answer**")
            st.write(case.get("answer") or "(none)")
            if case.get("path"):
                st.markdown("**Path:** " + " → ".join(case["path"]))
            if case.get("request_id"):
                st.caption(f"Request id {case['request_id']} (look it up in the Audit tab).")


def _results(client: CopilotClient, runs: list[dict]) -> None:
    st.markdown("#### Results")
    options = {r["run_id"]: r for r in runs}
    ids = list(options)
    if st.session_state.get("eval_pick") not in options:
        st.session_state["eval_pick"] = ids[0]
    run_id = st.selectbox("Run", ids, format_func=lambda i: ef.run_label(options[i]), key="eval_pick")
    run = options[run_id]
    _banner(*ef.status_banner(run))
    if run.get("error") and run["status"] not in ("stopped", "failed"):
        st.caption(run["error"])
    summary = _summary_for(client, run)
    if summary is None:
        if run["status"] in ef.ACTIVE:
            st.info("No summary yet. It appears once the run has written its first results.")
        else:
            st.info("No results were recorded for this run: it ended before any question finished. Resume it above, or pick another run.")
        return
    sections = [
        ("Overview", lambda: _overview_tab(run, summary)),
        ("Retrieval", lambda: _retrieval_tab(summary)),
        ("Answer quality", lambda: _answer_tab(summary)),
        ("Routing & external search", lambda: _routing_tab(summary)),
        ("Latency & cost", lambda: _system_tab(summary)),
        ("Baseline vs agentic", lambda: _comparison_tab(summary)),
    ]
    if ef.has_ablation(summary):
        sections.append(("Ablation", lambda: _ablation_tab(summary)))
    if ef.has_intervals(summary):
        sections.append(("Confidence intervals", lambda: _intervals_tab(summary)))
    if ef.has_attacks(summary):
        sections.append(("Attack success", lambda: _attacks_tab(summary)))
    sections.append(("Failed cases", lambda: _failed_tab(summary)))
    for tab, (_, draw) in zip(st.tabs([name for name, _ in sections]), sections):
        with tab:
            draw()
    with st.expander("Run history"):
        _table(ef.run_rows(runs))


def render_evaluation(client: CopilotClient, settings: Settings) -> None:
    """Draw the evaluation tab. Polls while a run is active when auto-refresh is on."""
    st.subheader("Evaluation")
    st.caption("Runs the labelled test set through the baseline RAG and the agentic workflow, then reports retrieval, answer, routing and cost metrics. Demo documents are synthetic.")
    try:
        runs = client.list_evaluation_runs()
    except ApiError as exc:
        _api_error(exc)
        return
    active = ef.active_run(runs)
    _controls(client, runs, active)
    if active:
        _active_panel(client, active)
    if runs:
        _results(client, runs)
    else:
        st.info("No evaluation runs yet. Start a small one above.")
    if active and not st.session_state.get("eval_pause_refresh", False):
        time.sleep(settings.ui_eval_refresh_s)
        st.rerun()
