"""Unit tests for ingestion.validators and ingestion.loaders."""
from __future__ import annotations

import io
import zipfile

import pytest
from fakes import make_docx_bytes, make_pdf_bytes

from copilot.ingestion.loaders import load_document
from copilot.ingestion.validators import IngestionError, ValidationFailed, safe_file_name, validate_upload


def test_valid_files_return_types(settings):
    assert validate_upload("a.txt", b"hello", settings) == "txt"
    assert validate_upload("a.docx", make_docx_bytes([("p", "x")]), settings) == "docx"
    assert validate_upload("a.pdf", make_pdf_bytes(["x"]), settings) == "pdf"


@pytest.mark.parametrize("name", ["a.exe", "a", "a.csv"])
def test_unsupported_extension_rejected(settings, name):
    with pytest.raises(ValidationFailed, match="Unsupported"):
        validate_upload(name, b"data", settings)


def test_empty_and_oversized_rejected(make_settings):
    s = make_settings(max_upload_mb=1)
    with pytest.raises(ValidationFailed, match="empty"):
        validate_upload("a.txt", b"", s)
    with pytest.raises(ValidationFailed, match="limit"):
        validate_upload("a.txt", b"x" * (1024 * 1024 + 1), s)


def test_signature_mismatch_rejected(settings):
    with pytest.raises(ValidationFailed, match="PDF"):
        validate_upload("a.pdf", b"not a pdf", settings)
    with pytest.raises(ValidationFailed, match="DOCX"):
        validate_upload("a.docx", b"not a zip", settings)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "x")
    with pytest.raises(ValidationFailed, match="document.xml"):
        validate_upload("a.docx", buf.getvalue(), settings)
    with pytest.raises(ValidationFailed, match="corrupt"):
        validate_upload("a.docx", b"PK\x03\x04garbage", settings)


def test_binary_txt_rejected(settings):
    with pytest.raises(ValidationFailed, match="binary"):
        validate_upload("a.txt", b"ab\x00cd", settings)


def test_safe_file_name():
    assert safe_file_name("../../etc/passwd.txt") == "passwd.txt"
    assert safe_file_name("C:\\x\\y.txt") == "y.txt"
    for bad in ["", "   ", "..", "a\x01.txt", "x" * 300 + ".txt"]:
        with pytest.raises(ValidationFailed):
            safe_file_name(bad)


def test_load_txt_utf8_and_cp1252():
    assert load_document("a.txt", "txt", "héllo".encode("utf-8")).pages[0].text == "héllo"
    assert load_document("a.txt", "txt", b"caf\xe9").pages[0].text == "café"


def test_load_docx_maps_headings_to_markers():
    data = make_docx_bytes([("title", "Guide"), ("h1", "Overview"), ("p", "Body text")])
    loaded = load_document("a.docx", "docx", data)
    text = loaded.pages[0].text
    assert "# Guide" in text and "## Overview" in text and "Body text" in text
    assert loaded.page_count is None


def test_load_pdf_preserves_page_numbers():
    loaded = load_document("a.pdf", "pdf", make_pdf_bytes(["First page about VPN", "Second page about MFA"]))
    assert [p.page_number for p in loaded.pages] == [1, 2]
    assert "VPN" in loaded.pages[0].text and "MFA" in loaded.pages[1].text
    assert loaded.page_count == 2


def test_corrupt_pdf_raises_ingestion_error():
    with pytest.raises(IngestionError):
        load_document("a.pdf", "pdf", b"%PDF-1.4\ngarbage that is not a pdf")
