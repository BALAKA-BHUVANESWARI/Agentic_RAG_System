"""Persistence for user feedback on answers (thumbs up/down), linked to the audit record."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from copilot.db.connection import connect
from copilot.logging_config import redact
from copilot.utils import utc_now_iso


class FeedbackStore:
    """Stores feedback rows. The comment is redacted and length-capped before saving."""

    def __init__(self, db_path: Path | str, max_comment_chars: int = 1000) -> None:
        self._db_path = Path(db_path)
        self._max_comment_chars = max_comment_chars

    def add(self, request_id: str, user_id: str | None, rating: Literal["positive", "negative"], comment: str | None = None) -> int:
        """Insert a feedback row and return its id. Raises sqlite3.IntegrityError for an unknown request_id."""
        clean = redact(comment.strip())[: self._max_comment_chars] if comment and comment.strip() else None
        with connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO feedback (request_id, user_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?)",
                (request_id, user_id, rating, clean, utc_now_iso()),
            )
            return int(cur.lastrowid)

    def for_request(self, request_id: str) -> list[dict]:
        """All feedback rows for a request (used by audit views and evaluation)."""
        with connect(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM feedback WHERE request_id = ? ORDER BY id", (request_id,)).fetchall()
        return [dict(r) for r in rows]
