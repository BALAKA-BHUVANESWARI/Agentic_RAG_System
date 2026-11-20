"""Unit tests for ingestion.metadata and ingestion.chunker."""
from __future__ import annotations

import pytest

from copilot.ingestion.chunker import build_segments, chunk_document
from copilot.ingestion.metadata import extract_metadata, split_header
from copilot.ingestion.models import LoadedDocument, PageText
from copilot.ingestion.validators import ValidationFailed

HEADER = (
    "SYNTHETIC DEMO DOCUMENT — NOT A REAL COMPANY POLICY\n"
    "Title: VPN Guide\nCategory: VPN\nDepartment: Network\nAccess-Level: Restricted\n"
    "Version: 3\nLast-Reviewed: 2025-02-14\n\n## 1. Overview\nBody here.\n"
)


def _txt(text: str, name: str = "vpn_guide.txt") -> LoadedDocument:
    return LoadedDocument(file_name=name, file_type="txt", pages=[PageText(text=text)])


def test_split_header_parses_and_strips():
    header, body = split_header(HEADER)
    assert header["title"] == "VPN Guide" and header["access-level"] == "Restricted"
    assert header["last-reviewed"] == "2025-02-14"
    assert body.lstrip().startswith("## 1. Overview")
    assert "SYNTHETIC" not in body and "Category:" not in body


def test_split_header_without_header_returns_text_unchanged():
    text = "Just a paragraph.\nCategory: not at top"
    assert split_header(text) == ({}, text)


def test_header_value_used_and_normalized():
    meta, loaded = extract_metadata(_txt(HEADER))
    assert (meta.title, meta.category, meta.department, meta.access_level) == ("VPN Guide", "vpn", "Network", "restricted")
    assert meta.last_reviewed == "2025-02-14" and meta.source_version == "3"
    assert "Access-Level" not in loaded.pages[0].text


def test_explicit_overrides_win_over_header():
    meta, _ = extract_metadata(_txt(HEADER), category="Security", access_level="public")
    assert meta.category == "security" and meta.access_level == "public"


def test_defaults_when_no_header_and_title_from_filename():
    meta, _ = extract_metadata(_txt("Plain body", name="Laptop_Setup-Guide.txt"))
    assert meta.title == "Laptop Setup Guide"
    assert (meta.category, meta.department, meta.access_level) == ("general", "IT", "internal")


def test_invalid_access_level_rejected():
    with pytest.raises(ValidationFailed):
        extract_metadata(_txt("body"), access_level="top-secret")


def test_build_segments_txt_sections():
    segs = build_segments(_txt("Intro text\n## A\nalpha\n## B\nbeta\n## Empty\n"))
    assert [(s.section, s.text) for s in segs] == [(None, "Intro text"), ("A", "alpha"), ("B", "beta")]


def test_build_segments_pdf_pages_and_blank_pages_skipped():
    loaded = LoadedDocument(
        file_name="a.pdf", file_type="pdf",
        pages=[PageText(page_number=1, text="one"), PageText(page_number=2, text="  "), PageText(page_number=3, text="three")],
    )
    segs = build_segments(loaded)
    assert [(s.page_number, s.text) for s in segs] == [(1, "one"), (3, "three")]


def test_chunk_document_metadata_ids_and_size(settings):
    meta, loaded = extract_metadata(_txt(HEADER + "word " * 300))
    chunks = chunk_document(
        document_id="doc1", document_name="vpn_guide.txt", version=2, uploaded_at="2026-01-01T00:00:00+00:00",
        segments=build_segments(loaded), metadata=meta, settings=settings,
    )
    assert len(chunks) > 2
    assert [c.chunk_id for c in chunks] == [f"doc1:v2:{i}" for i in range(len(chunks))]
    for c in chunks:
        assert len(c.text) <= settings.chunk_size
        assert (c.document_id, c.document_name, c.version, c.access_level) == ("doc1", "vpn_guide.txt", 2, "restricted")
        assert c.category == "vpn" and c.department == "Network" and c.uploaded_at
        assert c.section == "1. Overview"
    assert chunks[0].embedding_text.startswith("VPN Guide | 1. Overview\n")


def test_chunk_document_keeps_pdf_page_numbers(settings):
    loaded = LoadedDocument(
        file_name="a.pdf", file_type="pdf",
        pages=[PageText(page_number=1, text="alpha " * 10), PageText(page_number=2, text="beta " * 10)],
    )
    meta, loaded = extract_metadata(loaded)
    chunks = chunk_document(
        document_id="d", document_name="a.pdf", version=1, uploaded_at="t",
        segments=build_segments(loaded), metadata=meta, settings=settings,
    )
    assert [c.page_number for c in chunks] == [1, 2]


# ---- document status (superseded versions) ---------------------------------------------
def test_status_superseded_by_is_parsed_and_stripped_from_the_body():
    text = HEADER.replace("Last-Reviewed: 2025-02-14\n", "Last-Reviewed: 2025-02-14\nStatus: Superseded by VPN_Guide_v4.txt\n")
    meta, loaded = extract_metadata(_txt(text))
    assert (meta.status, meta.superseded_by) == ("superseded", "VPN_Guide_v4.txt")
    assert "Status:" not in loaded.pages[0].text and loaded.pages[0].text.lstrip().startswith("## 1. Overview")


def _status_of(text: str) -> tuple[str, str | None]:
    meta, _ = extract_metadata(_txt(text))
    return meta.status, meta.superseded_by


def test_status_defaults_to_current_and_accepts_explicit_values():
    assert _status_of(HEADER) == ("current", None)
    assert _status_of(HEADER.replace("Version: 3\n", "Version: 3\nStatus: current\n")) == ("current", None)
    assert _status_of(HEADER.replace("Version: 3\n", "Version: 3\nStatus: superseded\n")) == ("superseded", None)


def test_body_text_that_starts_with_status_is_not_consumed_as_a_header_line():
    header, body = split_header("Title: X\nStatus: green and healthy\n\nBody\n")
    assert header == {"title": "X"} and body.lstrip().startswith("Status: green and healthy")


def test_chunks_carry_the_document_status(settings):
    text = HEADER.replace("Version: 3\n", "Version: 3\nStatus: superseded by New.txt\n")
    meta, loaded = extract_metadata(_txt(text + "word " * 40))
    chunks = chunk_document(
        document_id="d", document_name="old.txt", version=1, uploaded_at="t",
        segments=build_segments(loaded), metadata=meta, settings=settings,
    )
    assert chunks and all((c.status, c.superseded_by) == ("superseded", "New.txt") for c in chunks)
