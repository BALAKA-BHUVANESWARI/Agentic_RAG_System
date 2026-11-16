"""Unit tests for copilot.logging_config."""
from __future__ import annotations

import json
import logging

from copilot.logging_config import (
    bind_request_id,
    get_request_id,
    redact,
    setup_logging,
)


def test_redacts_known_key_formats():
    text = "keys gsk_abcdefghij1234567 pcsk_abcdefghij12345 tvly-abcdefghij12345"
    out = redact(text)
    assert "gsk_" not in out and "pcsk_" not in out and "tvly-" not in out


def test_redacts_key_value_pairs_and_exact_secrets():
    out = redact("password=hunter2 and mytopsecret", secrets=["mytopsecret"])
    assert "hunter2" not in out and "mytopsecret" not in out


def test_request_id_binding_and_reset():
    assert get_request_id() == "-"
    with bind_request_id("req-123") as rid:
        assert rid == "req-123" and get_request_id() == "req-123"
    assert get_request_id() == "-"


def test_json_logs_contain_request_id_and_redact(capsys):
    setup_logging(level="INFO", json_logs=True, extra_secrets=["s3cr3t-value"])
    with bind_request_id("abc"):
        logging.getLogger("t").info("hello s3cr3t-value", extra={"doc": "x"})
    line = capsys.readouterr().out.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["request_id"] == "abc"
    assert data["doc"] == "x"
    assert "s3cr3t-value" not in line


def test_setup_logging_is_idempotent():
    setup_logging(level="INFO", json_logs=True, extra_secrets=[])
    setup_logging(level="INFO", json_logs=True, extra_secrets=[])
    marked = [h for h in logging.getLogger().handlers if getattr(h, "_copilot_handler", False)]
    assert len(marked) == 1
