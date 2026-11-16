"""Unit tests for copilot.config."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from copilot.config import PROJECT_ROOT


def test_defaults_match_project_requirements(make_settings):
    s = make_settings()
    assert s.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.embedding_dimension == 384
    assert s.max_rewrite_retries == 2
    assert s.max_validation_revisions == 1
    assert s.allowed_extensions == [".pdf", ".docx", ".txt"]


def test_relative_paths_are_anchored_to_project_root(make_settings):
    s = make_settings()
    assert s.sqlite_path.is_absolute()
    assert str(s.sqlite_path).startswith(str(PROJECT_ROOT))


def test_env_vars_override_defaults(make_settings, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "9")
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    s = make_settings()
    assert s.retrieval_top_k == 9
    assert s.groq_model == "some-model"


def test_overlap_must_be_smaller_than_chunk_size(make_settings):
    with pytest.raises(ValidationError):
        make_settings(chunk_size=200, chunk_overlap=200)


def test_threshold_out_of_range_rejected(make_settings):
    with pytest.raises(ValidationError):
        make_settings(private_evidence_threshold=1.5)


def test_invalid_log_level_rejected(make_settings):
    with pytest.raises(ValidationError):
        make_settings(log_level="LOUD")


def test_extensions_are_normalized(make_settings):
    assert make_settings(allowed_extensions=["PDF", ".Txt"]).allowed_extensions == [".pdf", ".txt"]


def test_role_access_levels(make_settings):
    s = make_settings()
    assert "restricted" not in s.access_levels_for("employee")
    assert "restricted" in s.access_levels_for("admin")
    assert s.access_levels_for("unknown-role") == []


def test_default_role_must_exist(make_settings):
    with pytest.raises(ValidationError):
        make_settings(default_role="ghost")


def test_require_secret_raises_without_leaking(make_settings):
    s = make_settings()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        s.require_secret("groq")


def test_secret_not_exposed_in_repr(make_settings):
    s = make_settings(groq_api_key="gsk_supersecretvalue123456")
    assert "supersecretvalue" not in repr(s)
    assert s.require_secret("groq") == "gsk_supersecretvalue123456"
    assert s.secret_values() == ["gsk_supersecretvalue123456"]


def test_get_settings_is_cached():
    from copilot.config import get_settings

    assert get_settings() is get_settings()


def test_max_upload_bytes(make_settings):
    assert make_settings(max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024


def test_paths_are_path_objects(make_settings):
    assert isinstance(make_settings().kb_documents_dir, Path)
