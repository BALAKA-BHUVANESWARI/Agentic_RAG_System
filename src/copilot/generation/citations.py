"""Resolve markers to real citations and render the user-facing answer."""
from __future__ import annotations

from copilot.generation.models import Citation, EvidenceItem

EXTERNAL_WARNING = (
    "External information: This response uses public vendor information and may not "
    "represent your company's internal IT policy."
)


def build_citations(evidence: list[EvidenceItem], used_refs: list[int]) -> list[Citation]:
    """Citations for the evidence actually cited in the answer, in citation order."""
    by_ref = {e.ref: e for e in evidence}
    return [
        Citation(
            ref=r, kind=by_ref[r].kind, label=by_ref[r].label, location=by_ref[r].location,
            url=by_ref[r].url, chunk_id=by_ref[r].chunk_id, document_id=by_ref[r].document_id,
            page_number=by_ref[r].page_number, section=by_ref[r].section,
            superseded=by_ref[r].status == "superseded", superseded_by=by_ref[r].superseded_by,
        )
        for r in used_refs
        if r in by_ref
    ]


def superseded_notice(citations: list[Citation]) -> str | None:
    """Fixed notice for answers that cite a document a newer one has replaced (None when none does).

    Added by code, not by the model, so the label never depends on what the LLM wrote.
    """
    old = [c for c in citations if c.superseded]
    if not old:
        return None
    parts = []
    for c in old:
        by = f" by {c.superseded_by}" if c.superseded_by else " by a newer document"
        parts.append(f"[{c.ref}] {c.label} has been superseded{by}")
    return "Outdated source: " + "; ".join(parts) + ". It is cited only for the earlier rule, not as current policy."


def conflict_explained_by_status(citations: list[Citation]) -> bool:
    """True when the answer cites exactly ONE current document plus a superseded one.

    A conflict between a current document and the older one it replaced is communicated by code:
    ``superseded_notice`` and ``render_sources`` label the older source, so the reader is told which
    value applies now. When two or more different current documents are cited, a conflict between
    them is not explained by status, so the model's judgement is still needed.
    """
    private = [c for c in citations if c.kind == "private"]
    current = {c.document_id or c.label for c in private if not c.superseded}
    return len(current) == 1 and any(c.superseded for c in private)


def render_sources(citations: list[Citation]) -> str:
    """Plain-text 'Sources' block built from real evidence (ASCII only)."""
    lines = ["Sources:"]
    for c in citations:
        detail = c.url if c.kind == "external" and c.url else c.location
        flag = " (superseded)" if c.superseded else ""
        lines.append(f"[{c.ref}] {c.label} - {detail}{flag}")
    return "\n".join(lines)


def render_answer(body: str, citations: list[Citation]) -> str:
    """Final text: external warning and superseded notice (when they apply), body, then Sources."""
    parts = []
    if any(c.kind == "external" for c in citations):
        parts.append(EXTERNAL_WARNING)
    notice = superseded_notice(citations)
    if notice:
        parts.append(notice)
    parts.append(body.strip())
    parts.append(render_sources(citations))
    return "\n\n".join(parts)
