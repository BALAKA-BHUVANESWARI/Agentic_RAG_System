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
