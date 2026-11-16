"""SQLite schema for documents, ingestion history, audit log, feedback and eval runs.

Standard library only. ``init_db`` is idempotent (``CREATE TABLE IF NOT EXISTS``).
Repositories that read/write these tables arrive in later phases.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from copilot.db.connection import connect

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per logical document (document_id = sha256 of the normalized name).
CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    document_name   TEXT NOT NULL UNIQUE,
    content_hash    TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'general',
    department      TEXT NOT NULL DEFAULT 'IT',
    access_level    TEXT NOT NULL DEFAULT 'internal'
                    CHECK (access_level IN ('public', 'internal', 'restricted')),
    version         INTEGER NOT NULL DEFAULT 1,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'indexed', 'failed')),
    file_type       TEXT,
    file_size_bytes INTEGER,
    page_count      INTEGER,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    uploaded_by     TEXT,
    uploaded_at     TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    error           TEXT
);

-- History of every ingestion attempt (new versions, re-indexing, failures).
CREATE TABLE IF NOT EXISTS ingestion_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    version       INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL
                  CHECK (status IN ('pending', 'processing', 'indexed', 'failed')),
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT
);

-- Full request lifecycle. JSON columns hold lists/dicts serialized as text.
CREATE TABLE IF NOT EXISTS audit_log (
    request_id                TEXT PRIMARY KEY,
    session_id                TEXT,
    user_id                   TEXT,
    role                      TEXT,
    mode                      TEXT NOT NULL DEFAULT 'agentic'
                              CHECK (mode IN ('agentic', 'baseline')),
    question                  TEXT NOT NULL,
    intent                    TEXT,
    retrieved_document_ids    TEXT,
    retrieved_chunk_ids       TEXT,
    rewritten_query           TEXT,
    retrieval_scores          TEXT,
    grading_results           TEXT,
    external_search_triggered INTEGER NOT NULL DEFAULT 0,
    external_urls             TEXT,
    final_answer              TEXT,
    citations                 TEXT,
    validation_result         TEXT,
    abstained                 INTEGER NOT NULL DEFAULT 0,
    abstention_reason         TEXT,
    trace                     TEXT,
    execution_time_ms         REAL,
    error                     TEXT,
    created_at                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL REFERENCES audit_log(request_id) ON DELETE CASCADE,
    user_id     TEXT,
    rating      TEXT NOT NULL CHECK (rating IN ('positive', 'negative')),
    comment     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id        TEXT PRIMARY KEY,
    mode          TEXT NOT NULL CHECK (mode IN ('agentic', 'baseline', 'both')),
    dataset_path  TEXT NOT NULL,
    num_cases     INTEGER NOT NULL DEFAULT 0,
    metrics       TEXT,
    results_path  TEXT,
    metadata      TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_status     ON documents(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_document   ON ingestion_records(document_id);
CREATE INDEX IF NOT EXISTS idx_audit_user           ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_created        ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_request     ON feedback(request_id);
"""

EXPECTED_TABLES = frozenset(
    {"schema_meta", "documents", "ingestion_records", "audit_log", "feedback", "eval_runs"}
)


def init_db(db_path: Path | str) -> None:
    """Create all tables/indexes if missing and record the schema version."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )


def list_tables(db_path: Path | str) -> set[str]:
    """Return the names of user tables present in the database."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return {row["name"] for row in rows}


def get_schema_version(db_path: Path | str) -> int | None:
    """Return the stored schema version, or None if the DB is not initialised."""
    try:
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row["value"]) if row else None
