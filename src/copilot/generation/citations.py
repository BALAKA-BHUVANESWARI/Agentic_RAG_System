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
        )
        for r in used_refs
        if r in by_ref
    ]


def render_sources(citations: list[Citation]) -> str:
    """Plain-text 'Sources' block built from real evidence (ASCII only)."""
    lines = ["Sources:"]
    for c in citations:
        detail = c.url if c.kind == "external" and c.url else c.location
        lines.append(f"[{c.ref}] {c.label} - {detail}")
    return "\n".join(lines)


def render_answer(body: str, citations: list[Citation]) -> str:
    """Final text: external warning (if any external source is cited), body, then Sources."""
    parts = []
    if any(c.kind == "external" for c in citations):
        parts.append(EXTERNAL_WARNING)
    parts.append(body.strip())
    parts.append(render_sources(citations))
    return "\n\n".join(parts)
