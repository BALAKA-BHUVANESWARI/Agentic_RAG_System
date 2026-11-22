"""The settings snapshot saved with every evaluation run must show every setting that changes answers (docs/RUN_PLAN.md pre-flight)."""
from __future__ import annotations

from copilot.evaluation.system import settings_snapshot


def test_snapshot_records_the_per_document_chunk_cap(make_settings):
    assert settings_snapshot(make_settings())["retrieval_max_chunks_per_doc"] is None
    assert settings_snapshot(make_settings(retrieval_max_chunks_per_doc=3))["retrieval_max_chunks_per_doc"] == 3
