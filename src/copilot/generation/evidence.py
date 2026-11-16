"""Build numbered evidence for the LLM, with prompt-injection hardening."""
from __future__ import annotations

import re

from copilot.generation.models import EvidenceItem
from copilot.vectorstore.models import RetrievedChunk

_DELIMITER_TAG = re.compile(r"<\s*/?\s*(evidence|question|answer)\b", re.IGNORECASE)


def neutralize_delimiters(text: str) -> str:
    """Escape ``<evidence>``/``<question>`` tags so content cannot close or fake a block."""
    return _DELIMITER_TAG.sub(lambda m: m.group(0).replace("<", "&lt;"), text)


def _attr(value: str) -> str:
    return re.sub(r"[\"<>\n\r]", " ", value).strip()


def from_retrieved_chunks(chunks: list[RetrievedChunk]) -> list[EvidenceItem]:
    """Convert retrieved private chunks to evidence items numbered from 1."""
    return [
        EvidenceItem(
            ref=i, kind="private", label=c.document_name, location=c.location, text=c.text,
            chunk_id=c.chunk_id, document_id=c.document_id, page_number=c.page_number, section=c.section,
            last_reviewed=c.last_reviewed,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def format_evidence(items: list[EvidenceItem], max_chars: int) -> str:
    """Render evidence as delimited blocks. Content is data, never instructions."""
    blocks = []
    for item in items:
        text = neutralize_delimiters(item.text.strip())[:max_chars]
        extras = ""
        if item.last_reviewed:
            extras += f' reviewed="{_attr(item.last_reviewed)}"'
        if item.published_date:
            extras += f' published="{_attr(item.published_date)}"'
        if item.url:
            extras += f' url="{_attr(item.url)}"'
        header = (
            f'<evidence id="{item.ref}" type="{item.kind}" '
            f'source="{_attr(item.label)}" location="{_attr(item.location)}"{extras}>'
        )
        blocks.append(f"{header}\n{text}\n</evidence>")
    return "\n\n".join(blocks)
