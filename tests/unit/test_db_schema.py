"""Unit tests for copilot.db (SQLite schema)."""
from __future__ import annotations

import sqlite3

import pytest

from copilot.db.connection import connect
from copilot.db.schema import EXPECTED_TABLES, SCHEMA_VERSION, get_schema_version, init_db, list_tables

NOW = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "nested" / "test.db"
    init_db(path)
    return path


def test_init_creates_all_tables_and_parent_dir(db):
    assert list_tables(db) >= EXPECTED_TABLES


def test_init_is_idempotent_and_records_version(db):
    init_db(db)
    assert get_schema_version(db) == SCHEMA_VERSION


def test_schema_version_none_when_uninitialised(tmp_path):
    assert get_schema_version(tmp_path / "empty.db") is None


def _insert_doc(conn, **kw):
    row = dict(document_id="d1", document_name="a.txt", content_hash="h", access_level="internal",
               status="indexed", uploaded_at=NOW, updated_at=NOW)
    row.update(kw)
    conn.execute(
        "INSERT INTO documents(document_id, document_name, content_hash, access_level, status, uploaded_at, updated_at) "
        "VALUES (:document_id, :document_name, :content_hash, :access_level, :status, :uploaded_at, :updated_at)",
        row,
    )


def test_invalid_access_level_rejected(db):
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as conn:
            _insert_doc(conn, access_level="top-secret")


def test_invalid_status_rejected(db):
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as conn:
            _insert_doc(conn, status="done")


def test_document_name_unique(db):
    with connect(db) as conn:
        _insert_doc(conn)
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as conn:
            _insert_doc(conn, document_id="d2")


def test_ingestion_records_cascade_on_document_delete(db):
    with connect(db) as conn:
        _insert_doc(conn)
        conn.execute(
            "INSERT INTO ingestion_records(document_id, version, content_hash, status, started_at) "
            "VALUES ('d1', 1, 'h', 'indexed', ?)", (NOW,))
    with connect(db) as conn:
        conn.execute("DELETE FROM documents WHERE document_id = 'd1'")
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM ingestion_records").fetchone()["n"] == 0


def test_feedback_requires_existing_audit_row(db):
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as conn:
            conn.execute(
                "INSERT INTO feedback(request_id, rating, created_at) VALUES ('nope', 'positive', ?)", (NOW,))


def test_rollback_on_error(db):
    with pytest.raises(RuntimeError):
        with connect(db) as conn:
            _insert_doc(conn)
            raise RuntimeError("boom")
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"] == 0


# ---- version status columns (schema version 2) -----------------------------------------
def _document_columns(path):
    with connect(path) as conn:
        return {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}


def test_new_database_has_version_status_columns_and_rejects_unknown_values(db):
    assert {"lifecycle", "superseded_by"} <= _document_columns(db)
    with connect(db) as conn:
        _insert_doc(conn)
        assert tuple(conn.execute("SELECT lifecycle, superseded_by FROM documents").fetchone()) == ("current", None)
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as conn:
            conn.execute("UPDATE documents SET lifecycle = 'obsolete'")


_OLD_DOCUMENTS_TABLE = """
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY, document_name TEXT NOT NULL UNIQUE, content_hash TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general', department TEXT NOT NULL DEFAULT 'IT',
    access_level TEXT NOT NULL DEFAULT 'internal' CHECK (access_level IN ('public', 'internal', 'restricted')),
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'indexed', 'failed')),
    file_type TEXT, file_size_bytes INTEGER, page_count INTEGER, chunk_count INTEGER NOT NULL DEFAULT 0,
    uploaded_by TEXT, uploaded_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT
);
"""


def test_database_from_before_version_status_is_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "old.db"
    with connect(path) as conn:
        conn.executescript(_OLD_DOCUMENTS_TABLE)
        _insert_doc(conn, document_id="old1", document_name="old.txt")
    assert "lifecycle" not in _document_columns(path)
    init_db(path)
    init_db(path)  # running it again is harmless
    assert {"lifecycle", "superseded_by"} <= _document_columns(path)
    with connect(path) as conn:
        row = conn.execute("SELECT document_name, lifecycle, superseded_by FROM documents").fetchone()
    assert (row["document_name"], row["lifecycle"], row["superseded_by"]) == ("old.txt", "current", None)
    assert get_schema_version(path) == SCHEMA_VERSION == 2
