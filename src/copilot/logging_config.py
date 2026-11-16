"""Structured logging with per-request IDs and secret redaction.

Standard library only, so it can be imported before settings load. Every log line
carries ``request_id`` (from a ContextVar) and passes through a redaction filter so
API keys never reach logs. Full document text must never be logged by callers.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_REDACTED = "[REDACTED]"
_HANDLER_MARK = "_copilot_handler"
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),  # Groq
    re.compile(r"pcsk_[A-Za-z0-9_]{10,}"),  # Pinecone
    re.compile(r"tvly-[A-Za-z0-9\-]{10,}"),  # Tavily
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{10,}"),
]
_KV_PATTERN = re.compile(r"(?i)\b(api[_-]?key|token|password|secret)(\s*[=:]\s*)\S+")
# Attributes present on every LogRecord; anything else came from `extra=`.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def new_request_id() -> str:
    """Generate a short unique request identifier."""
    return uuid.uuid4().hex[:16]


def get_request_id() -> str:
    """Return the request ID bound to the current context ('-' if none)."""
    return request_id_var.get()


@contextmanager
def bind_request_id(request_id: str | None = None) -> Iterator[str]:
    """Bind a request ID to the current context for the duration of the block."""
    rid = request_id or new_request_id()
    token = request_id_var.set(rid)
    try:
        yield rid
    finally:
        request_id_var.reset(token)


def redact(text: str, secrets: list[str] | None = None) -> str:
    """Mask known API-key formats, ``key=value`` credentials and exact secret values."""
    for secret in secrets or []:
        if secret:
            text = text.replace(secret, _REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return _KV_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", text)


class RedactingFilter(logging.Filter):
    """Adds ``request_id`` to each record and redacts secrets from message/extras."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self._secrets = [s for s in (secrets or []) if s]

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.msg = redact(record.getMessage(), self._secrets)
        record.args = None
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_ATTRS and key != "request_id" and isinstance(value, str):
                record.__dict__[key] = redact(value, self._secrets)
        return True


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


_TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"


def setup_logging(
    level: str | None = None,
    json_logs: bool | None = None,
    extra_secrets: list[str] | None = None,
) -> None:
    """Configure root logging once (idempotent).

    With no arguments, values come from settings (imported lazily so this module
    stays usable on its own, e.g. in tests).
    """
    if level is None or json_logs is None or extra_secrets is None:
        from copilot.config import get_settings  # lazy import

        settings = get_settings()
        level = level or settings.log_level
        json_logs = settings.log_json if json_logs is None else json_logs
        extra_secrets = settings.secret_values() if extra_secrets is None else extra_secrets

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(RedactingFilter(extra_secrets))
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "pinecone"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Call :func:`setup_logging` once at process start."""
    return logging.getLogger(name)
