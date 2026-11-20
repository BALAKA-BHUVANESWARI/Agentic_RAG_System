"""Segmenting and chunking: preserves page numbers (PDF) and section headings (DOCX/TXT)."""
from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from copilot.config import Settings
from copilot.ingestion.ids import make_chunk_id
from copilot.ingestion.models import Chunk, DocumentMetadata, LoadedDocument, Segment

_HEADING = re.compile(r"^#{1,3}\s+(.+?)\s*$")


def _split_by_headings(text: str, page_number: int | None) -> list[Segment]:
    segments: list[Segment] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            segments.append(Segment(page_number=page_number, section=section, text=body))
        buffer.clear()

    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            flush()
            section = match.group(1)
        else:
            buffer.append(line)
    flush()
    return segments


def build_segments(loaded: LoadedDocument) -> list[Segment]:
    """Turn loaded pages into located segments (one per PDF page; one per heading otherwise)."""
    segments: list[Segment] = []
    for page in loaded.pages:
        if not page.text.strip():
            continue
        if loaded.file_type == "pdf":
            segments.append(Segment(page_number=page.page_number, section=None, text=page.text.strip()))
        else:
            segments.extend(_split_by_headings(page.text, page.page_number))
    return segments


def chunk_document(
    *,
    document_id: str,
    document_name: str,
    version: int,
    uploaded_at: str,
    segments: list[Segment],
    metadata: DocumentMetadata,
    settings: Settings,
) -> list[Chunk]:
    """Split segments into overlapping chunks carrying all required metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        strip_whitespace=True,
    )
    chunks: list[Chunk] = []
    for segment in segments:
        for piece in splitter.split_text(segment.text):
            if not piece.strip():
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document_id, version, index),
                    document_id=document_id,
                    document_name=document_name,
                    title=metadata.title,
                    chunk_index=index,
                    text=piece,
                    category=metadata.category,
                    department=metadata.department,
                    access_level=metadata.access_level,
                    page_number=segment.page_number,
                    section=segment.section,
                    uploaded_at=uploaded_at,
                    version=version,
                    last_reviewed=metadata.last_reviewed,
                    status=metadata.status,
                    superseded_by=metadata.superseded_by,
                )
            )
    return chunks
