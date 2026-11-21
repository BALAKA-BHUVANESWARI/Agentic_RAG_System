"""SQLite audit logging of the full request lifecycle.

Secrets are redacted from the stored question, answer and trace, and the logger never raises:
an audit failure must not break answering (it is logged instead).
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from copilot.db.connection import connect
from copilot.db.schema import init_db
from copilot.logging_config import get_logger, redact
from copilot.search.sanitizer import redact_credentials, redact_sensitive
from copilot.utils import utc_now_iso

log = get_logger(__name__)


# Keys whose values are text a person typed (or a rewrite of it): they get the full rule set.
_USER_TEXT_KEYS = frozenset({"query", "rewritten_query", "question"})


def _safe_text(text: str | None) -> str | None:
    """Redact text a person typed: credentials, API keys and long opaque tokens."""
    return None if text is None else redact(redact_sensitive(text))


def _safe_generated(text: str | None) -> str | None:
    """Redact credentials in generated text without touching long URLs or ids."""
    return None if text is None else redact(redact_credentials(text))


def _redact_tree(value: Any, key: str | None = None) -> Any:
    """Redact every string inside a JSON-like structure before it is stored.

    Values under ``_USER_TEXT_KEYS`` get the full rule set; all other strings get the credential
    rules only, so chunk ids and URLs in the trace stay readable.
    """
    if isinstance(value, str):
        return _safe_text(value) if key in _USER_TEXT_KEYS else _safe_generated(value)
    if isinstance(value, dict):
        return {k: _redact_tree(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_tree(v, key) for v in value]
    return value


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


class AuditRecord(BaseModel):
    """An audit row with JSON columns parsed."""

    request_id: str
    session_id: str | None = None
    user_id: str | None = None
    role: str | None = None
    mode: str = "agentic"
    question: str
    intent: str | None = None
    retrieved_document_ids: list[str] = []
    retrieved_chunk_ids: list[str] = []
    rewritten_query: str | None = None
    retrieval_scores: list[dict[str, Any]] = []
    grading_results: dict[str, Any] = {}
    external_search_triggered: bool = False
    external_urls: list[str] = []
    final_answer: str | None = None
    citations: list[dict[str, Any]] = []
    validation_result: dict[str, Any] = {}
    abstained: bool = False
    abstention_reason: str | None = None
    trace: list[dict[str, Any]] = []
    execution_time_ms: float | None = None
    error: str | None = None
    created_at: str = ""


_JSON_FIELDS = {
    "retrieved_document_ids": [], "retrieved_chunk_ids": [], "retrieval_scores": [], "grading_results": {},
    "external_urls": [], "citations": [], "validation_result": {}, "trace": [],
}


class AuditLogger:
    """Writes and reads ``audit_log`` rows."""

    def __init__(self, db_path: Any) -> None:
        self._db_path = db_path
        init_db(db_path)

    def record_state(self, state: dict[str, Any], *, execution_time_ms: float, mode: Literal["agentic", "baseline"] = "agentic") -> None:
        """Persist a finished graph state. Never raises."""
        try:
            self._insert(state, execution_time_ms, mode)
        except Exception as exc:  # noqa: BLE001 - audit must never break the request
            log.error("Audit write failed: %s", type(exc).__name__)

    def _insert(self, state: dict[str, Any], execution_time_ms: float, mode: str) -> None:
        answer = state.get("answer") or {}
        attempts = state.get("retrieval_attempts") or []
        doc_ids = sorted({d for a in attempts for d in a.get("document_ids", [])})
        chunk_ids = sorted({c for a in attempts for c in a.get("chunk_ids", [])})
        scores = [
            {"attempt": a.get("attempt"), "query": _safe_text(a.get("query")), **s}
            for a in attempts for s in a.get("scores", [])
        ]
        external = state.get("external_results") or []
        grading = {
            "private": state.get("private_grade") or {},
            "external": state.get("external_grade") or {},
            "classification": state.get("classification") or {},
        }
        rewrites = state.get("rewritten_queries") or []
        row = {
            "request_id": state["request_id"],
            "session_id": state.get("session_id"),
            "user_id": state.get("user_id"),
            "role": state.get("role"),
            "mode": mode,
            "question": _safe_text(state.get("question", "")),
            "intent": (state.get("classification") or {}).get("intent"),
            "retrieved_document_ids": _dumps(doc_ids),
            "retrieved_chunk_ids": _dumps(chunk_ids),
            "rewritten_query": _safe_text(rewrites[-1]) if rewrites else None,
            "retrieval_scores": _dumps(scores),
            "grading_results": _dumps(_redact_tree(grading)),
            "external_search_triggered": int(bool(state.get("external_triggered"))),
            "external_urls": _dumps([r.get("url") for r in external if r.get("url")]),
            "final_answer": _safe_generated(answer.get("rendered")),
            "citations": _dumps(answer.get("citations") or []),
            "validation_result": _dumps(_redact_tree(state.get("validation") or {})),
            "abstained": int(bool(answer.get("abstained"))),
            "abstention_reason": answer.get("abstention_reason"),
            "trace": _dumps(_redact_tree(state.get("trace") or [])),
            "execution_time_ms": execution_time_ms,
            "error": _safe_text(state.get("fatal_error") or answer.get("error")),
            "created_at": utc_now_iso(),
        }
        cols = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with connect(self._db_path) as conn:
            conn.execute(f"INSERT OR REPLACE INTO audit_log ({cols}) VALUES ({placeholders})", row)

    def get(self, request_id: str) -> AuditRecord | None:
        """Fetch one audit record by request ID."""
        with connect(self._db_path) as conn:
            row = conn.execute("SELECT * FROM audit_log WHERE request_id = ?", (request_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        for key, default in _JSON_FIELDS.items():
            data[key] = json.loads(data[key]) if data.get(key) else default
        data["external_search_triggered"] = bool(data["external_search_triggered"])
        data["abstained"] = bool(data["abstained"])
        return AuditRecord(**data)
