"""Unit tests for ingestion.ids and ingestion.cleaner."""
from __future__ import annotations

from copilot.ingestion.cleaner import clean_text
from copilot.ingestion.ids import make_chunk_id, make_content_hash, make_document_id, normalize_name


def test_document_id_is_stable_and_ignores_case_and_directories():
    a = make_document_id("VPN_Guide.txt")
    assert a == make_document_id("c:\\docs\\vpn_guide.TXT")
    assert a == make_document_id("/tmp/x/VPN_Guide.txt")
    assert len(a) == 64 and a != make_document_id("other.txt")


def test_normalize_name():
    assert normalize_name("  Dir/My File.TXT ") == "my file.txt"


def test_chunk_id_format():
    assert make_chunk_id("abc", 2, 5) == "abc:v2:5"


def test_content_hash_changes_with_text_or_access_metadata():
    base = dict(category="vpn", department="IT", access_level="internal")
    h = make_content_hash("hello", **base)
    assert h == make_content_hash("hello", **base)
    assert h != make_content_hash("hello!", **base)
    assert h != make_content_hash("hello", **{**base, "access_level": "restricted"})


def test_clean_text_normalizes_whitespace_and_line_endings():
    assert clean_text("a\r\nb   c\t\td\r\n\r\n\r\n\r\ne") == "a\nb c d\n\ne"


def test_clean_text_removes_control_chars_and_bom():
    assert clean_text("\ufeffhel\x00lo") == "hello"


def test_clean_text_rejoins_hyphenated_words_only_before_lowercase():
    assert clean_text("net-\nwork") == "network"
    assert clean_text("VPN-\nClient") == "VPN-\nClient"


def test_clean_text_keeps_headings_and_list_structure():
    text = "## Title\n1. one\n2. two"
    assert clean_text(text) == text
