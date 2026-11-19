"""Chat page: conversation, citations, external warning, feedback and the decision trace."""
from __future__ import annotations

import streamlit as st

from copilot.ui import theme
from copilot.ui.api_client import ApiError, CopilotClient
from copilot.ui.formatting import (format_citation, format_ms, path_text, split_external_warning, timeline_rows,
                                   trace_sections)


def _render_trace(r: dict, key: str) -> None:
    """Decision trace built from the real graph events returned by the API."""
    st.markdown(theme.route_html(r["path"]), unsafe_allow_html=True)
    st.caption(f"Path: {path_text(r['path'])}")
    cols = st.columns(5)
    cols[0].metric("Total time", format_ms(r.get("latency_ms")))
    cols[1].metric("LLM calls", r.get("llm_calls", 0))
    cols[2].metric("Tokens", r.get("tokens", 0))
    cols[3].metric("Rewrites", r.get("retry_count", 0))
    cols[4].metric("Tavily calls", r.get("tavily_calls", 0))
    for sec in trace_sections(r):
        with st.expander(f"{sec.title} — {sec.summary}"):
            st.json(sec.data, expanded=True)
    with st.expander("Execution timeline (every graph node that ran)"):
        st.table(timeline_rows(r.get("trace", [])))
    st.caption(f"Request ID: `{r['request_id']}`")


def _render_feedback(client: CopilotClient, r: dict, idx: int) -> None:
    given = st.session_state.setdefault("feedback", {})
    rid = r["request_id"]
    if rid in given:
        st.caption(f"Thanks, feedback recorded ({given[rid]}).")
        return
    up, down, _ = st.columns([2, 2, 3])
    for col, rating, label in ((up, "positive", "👍 Helpful"), (down, "negative", "👎 Not helpful")):
        if col.button(label, key=f"fb_{rating}_{idx}"):
            try:
                client.feedback(rid, rating)
                given[rid] = rating
                st.rerun()
            except ApiError as exc:
                st.error(f"Could not save feedback: {exc.detail}")


def _render_assistant(client: CopilotClient, m: dict, idx: int) -> None:
    with st.chat_message("assistant"):
        if "error" in m:
            st.error(m["error"])
            return
        r = m["response"]
        st.markdown(theme.verdict_html(r["abstained"], r["used_external"], (r.get("validation") or {}).get("passed")), unsafe_allow_html=True)
        warning, body = split_external_warning(r["answer"], r["used_external"])
        if warning:
            st.warning(warning, icon="🌐")
        (st.info if r["abstained"] else st.markdown)(body)
        if r["citations"]:
            with st.expander(f"Source details ({len(r['citations'])})"):
                for c in r["citations"]:
                    st.markdown(format_citation(c))
                    if c.get("chunk_id"):
                        st.caption(f"chunk `{c['chunk_id']}`")
        _render_feedback(client, r, idx)
        if st.toggle("Show decision trace", key=f"trace_{idx}"):
            _render_trace(r, f"t{idx}")


def render_chat(client: CopilotClient, allow_external: bool) -> None:
    """Draw the chat page and handle a new question."""
    st.header("Ask the IT desk")
    st.markdown('<p class="lede">Company IT documents are searched first. Public web search is used only when they are not enough, and is always labelled.</p>', unsafe_allow_html=True)
    messages: list[dict] = st.session_state.setdefault("messages", [])
    for i, m in enumerate(messages):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            _render_assistant(client, m, i)

    if not messages:
        st.markdown('<div class="starters-title">Try a question</div>', unsafe_allow_html=True)
        for j, text in enumerate(theme.STARTERS):
            if st.button(text, key=f"starter_{j}"):
                st.session_state["starter"] = text
                st.rerun()
    question = st.chat_input("Ask an IT support question…", key="question") or st.session_state.pop("starter", None)
    if not question:
        return
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Searching company documents…"):
        try:
            resp = client.chat(question, session_id=st.session_state.setdefault("session_id", None), external_search_allowed=allow_external)
            messages.append({"role": "assistant", "response": resp})
        except ApiError as exc:
            messages.append({"role": "assistant", "error": f"{exc.detail}" + (f" (request {exc.request_id})" if exc.request_id else "")})
    st.rerun()
