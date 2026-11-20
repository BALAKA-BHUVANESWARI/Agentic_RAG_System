"""Pure helpers that turn API responses into display data (no Streamlit import, unit-testable)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from copilot.generation.citations import EXTERNAL_WARNING


@dataclass(frozen=True)
class TraceSection:
    """One block of the decision-trace panel."""

    title: str
    summary: str
    data: Any


def split_external_warning(answer: str, used_external: bool) -> tuple[str | None, str]:
    """Separate the mandatory external-information warning from the answer body."""
    if used_external and answer.startswith(EXTERNAL_WARNING):
        return EXTERNAL_WARNING, answer[len(EXTERNAL_WARNING):].lstrip()
    return None, answer


def format_citation(c: dict) -> str:
    """One markdown line for a citation. External sources become links; company documents show page/section."""
    head = f"**[{c['ref']}]** "
    if c.get("kind") == "external":
        return head + (f"[{c['label']}]({c['url']})" if c.get("url") else c["label"]) + " · _public source_"
    return head + f"{c['label']} · {c.get('location') or 'document'}"


def format_ms(ms: float | None) -> str:
    """Human-friendly duration."""
    if ms is None:
        return "-"
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def path_text(path: list[str]) -> str:
    """The executed node path as one line."""
    return " → ".join(path)


def timeline_rows(trace: list[dict]) -> list[dict]:
    """Rows for the execution timeline table (one per real graph event)."""
    return [
        {"#": i, "node": e["node"], "time": format_ms(e.get("duration_ms")), "what happened": e.get("summary", ""), "decision": e.get("decision") or ""}
        for i, e in enumerate(trace, 1)
    ]


def trace_sections(r: dict) -> list[TraceSection]:
    """Build the labelled decision-trace sections from a /chat response (all values come from the run)."""
    out: list[TraceSection] = []
    c = r.get("classification") or {}
    if c:
        out.append(TraceSection("Query classification", f"intent = {c.get('intent')} · sensitive = {c.get('contains_sensitive')} · needs current public info = {c.get('needs_current_public_info')}", c))

    attempts = r.get("retrieval_attempts") or []
    if attempts:
        chunks = sum(len(a.get("chunk_ids", [])) for a in attempts)
        out.append(TraceSection("Private retrieval", f"{len(attempts)} attempt(s), {chunks} chunk(s) retrieved", attempts))

    g = r.get("private_grade") or {}
    if g:
        verdict = "sufficient" if g.get("effective_sufficient") else "insufficient"
        out.append(TraceSection("Private evidence grading", f"{verdict} · confidence {g.get('confidence')} · action {g.get('recommended_action')}", g))

    rewrites = r.get("rewritten_queries") or []
    out.append(TraceSection("Query rewriting", f"{len(rewrites)} rewrite(s), retry count {r.get('retry_count', 0)}" if rewrites else "not needed",
                            {"rewritten_queries": rewrites, "retry_count": r.get("retry_count", 0)}))

    if r.get("external_search_triggered"):
        ext = f"public search ran ({len(r.get('external_results') or [])} result(s)); query: {r.get('external_query')}"
    elif r.get("external_blocked_reason"):
        ext = f"not used: {r['external_blocked_reason']}"
    else:
        ext = "not needed: company documents were enough, or nothing could justify a public search"
    out.append(TraceSection("External search decision", ext, {"triggered": r.get("external_search_triggered", False), "query": r.get("external_query"),
                                                              "blocked_reason": r.get("external_blocked_reason"), "tavily_calls": r.get("tavily_calls", 0),
                                                              "results": r.get("external_results") or []}))
    if r.get("external_grade"):
        eg = r["external_grade"]
        out.append(TraceSection("External evidence grading", f"{len(eg.get('accepted_refs') or [])} source(s) accepted · confidence {eg.get('confidence')}", eg))

    v = r.get("validation")
    if v:
        out.append(TraceSection("Final grounding validation", "passed" if v.get("passed") else "failed", v))

    out.append(TraceSection("Execution", f"{format_ms(r.get('latency_ms'))} · {r.get('tokens', 0)} tokens · {r.get('llm_calls', 0)} LLM calls · {r.get('tavily_calls', 0)} Tavily call(s)",
                            {"latency_ms": r.get("latency_ms"), "tokens": r.get("tokens"), "llm_calls": r.get("llm_calls"), "tavily_calls": r.get("tavily_calls")}))
    return out


def version_status_text(doc: dict) -> str:
    """'Current', 'Superseded' or 'Superseded by <document>' (the lifecycle fields may be missing in older responses)."""
    if doc.get("lifecycle") != "superseded":
        return "Current"
    return f"Superseded by {doc['superseded_by']}" if doc.get("superseded_by") else "Superseded"


def documents_table(docs: list[dict]) -> list[dict]:
    """Rows for the document table. Admin-only columns appear only when the API supplied them."""
    rows = []
    for d in docs:
        row = {"name": d["document_name"], "category": d["category"], "department": d["department"], "access": d["access_level"], "version": d["version"],
               "version status": version_status_text(d),
               "status": d["status"], "chunks": d["chunk_count"], "pages": d.get("page_count"), "updated": d.get("updated_at", "")[:19].replace("T", " ")}
        if d.get("uploaded_by") is not None:
            row["uploaded by"] = d["uploaded_by"]
        if d.get("error"):
            row["error"] = d["error"]
        rows.append(row)
    return rows
