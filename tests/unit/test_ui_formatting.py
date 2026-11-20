"""Unit tests for the UI's pure formatting helpers."""
from __future__ import annotations

from copilot.generation.citations import EXTERNAL_WARNING
from copilot.ui.formatting import (documents_table, format_citation, format_ms, path_text, split_external_warning,
                                   timeline_rows, trace_sections)


def test_external_warning_is_split_only_when_external_was_used():
    answer = f"{EXTERNAL_WARNING}\n\nCheck the health page [1]."
    assert split_external_warning(answer, True) == (EXTERNAL_WARNING, "Check the health page [1].")
    assert split_external_warning(answer, False) == (None, answer)
    assert split_external_warning("Plain answer [1].", True) == (None, "Plain answer [1].")


def test_citation_formatting():
    priv = format_citation({"ref": 1, "kind": "private", "label": "VPN_Connection_Guide.txt", "location": "section '3. Connecting from home'"})
    ext = format_citation({"ref": 2, "kind": "external", "label": "Service health", "location": "x", "url": "https://learn.microsoft.com/a"})
    assert priv == "**[1]** VPN_Connection_Guide.txt · section '3. Connecting from home'"
    assert "[Service health](https://learn.microsoft.com/a)" in ext and "public source" in ext


def test_small_helpers():
    assert format_ms(None) == "-" and format_ms(250) == "250 ms" and format_ms(2500) == "2.5 s"
    assert path_text(["a", "b"]) == "a → b"
    rows = timeline_rows([{"node": "query_classifier", "duration_ms": 12.3, "summary": "s", "decision": None}])
    assert rows == [{"#": 1, "node": "query_classifier", "time": "12 ms", "what happened": "s", "decision": ""}]


def _response(**kw):
    base = dict(classification={"intent": "internal_it", "contains_sensitive": False, "needs_current_public_info": False},
                retrieval_attempts=[{"attempt": 0, "chunk_ids": ["a", "b"]}], private_grade={"effective_sufficient": True, "confidence": 0.9, "recommended_action": "answer"},
                rewritten_queries=[], retry_count=0, external_search_triggered=False, external_results=[], validation={"passed": True},
                latency_ms=1500, tokens=100, llm_calls=4, tavily_calls=0)
    base.update(kw)
    return base


def test_trace_sections_scenario_a_says_no_external_search():
    titles = {s.title: s for s in trace_sections(_response())}
    assert list(titles) == ["Query classification", "Private retrieval", "Private evidence grading", "Query rewriting", "External search decision", "Final grounding validation", "Execution"]
    assert "2 chunk(s)" in titles["Private retrieval"].summary and "sufficient" in titles["Private evidence grading"].summary
    assert titles["Query rewriting"].summary == "not needed" and titles["External search decision"].summary.startswith("not needed")
    assert titles["Execution"].summary == "1.5 s · 100 tokens · 4 LLM calls · 0 Tavily call(s)"


def test_trace_sections_external_path_and_blocked_reason():
    ran = {s.title: s for s in trace_sections(_response(
        private_grade={"effective_sufficient": False, "confidence": 0.2, "recommended_action": "external_search"}, rewritten_queries=["q2"], retry_count=1,
        external_search_triggered=True, external_query="teams outage", external_results=[{"url": "u"}], external_grade={"accepted_refs": [1], "confidence": 0.9}, tavily_calls=1))}
    assert "insufficient" in ran["Private evidence grading"].summary and ran["Query rewriting"].summary.startswith("1 rewrite")
    assert "teams outage" in ran["External search decision"].summary and "1 source(s) accepted" in ran["External evidence grading"].summary
    blocked = {s.title: s for s in trace_sections(_response(external_blocked_reason="request forbids public search"))}
    assert blocked["External search decision"].summary == "not used: request forbids public search"


def test_documents_table_shows_admin_columns_only_when_present():
    base = dict(document_name="a.txt", category="c", department="IT", access_level="internal", version=1, status="indexed", chunk_count=3, page_count=None, updated_at="2026-11-16T10:00:00+00:00")
    emp = documents_table([{**base, "uploaded_by": None, "error": None}])[0]
    adm = documents_table([{**base, "uploaded_by": "root", "error": "boom"}])[0]
    assert "uploaded by" not in emp and "error" not in emp and emp["updated"] == "2026-11-16 10:00:00"
    assert adm["uploaded by"] == "root" and adm["error"] == "boom"


def test_documents_table_shows_the_version_status():
    base = dict(document_name="a.txt", category="c", department="IT", access_level="internal", version=1, status="indexed", chunk_count=3, page_count=None, updated_at="2026-11-16T10:00:00+00:00")
    rows = documents_table([base, {**base, "lifecycle": "superseded", "superseded_by": "b.txt"}, {**base, "lifecycle": "superseded"}])
    assert [r["version status"] for r in rows] == ["Current", "Superseded by b.txt", "Superseded"]
