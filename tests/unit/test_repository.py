"""Unit tests for DocumentRepository."""
from __future__ import annotations

from copilot.db.repository import DocumentRecord


def _rec(i: int = 1, **kw) -> DocumentRecord:
    base = dict(document_id=f"d{i}", document_name=f"doc{i}.txt", content_hash=f"h{i}", status="indexed")
    base.update(kw)
    return DocumentRecord(**base)


def test_upsert_get_and_update(repo):
    repo.upsert(_rec(1))
    got = repo.get("d1")
    assert got.document_name == "doc1.txt" and got.uploaded_at and got.updated_at
    repo.upsert(_rec(1, version=2, content_hash="new"))
    assert repo.get("d1").version == 2 and repo.get("d1").content_hash == "new"
    assert repo.get("missing") is None


def test_get_by_hash_only_returns_indexed(repo):
    repo.upsert(_rec(1, status="failed"))
    assert repo.get_by_hash("h1") is None
    repo.upsert(_rec(2))
    assert repo.get_by_hash("h2").document_id == "d2"


def test_set_status_updates_fields(repo):
    repo.upsert(_rec(1, status="processing"))
    repo.set_status("d1", "indexed", chunk_count=7)
    assert (repo.get("d1").status, repo.get("d1").chunk_count) == ("indexed", 7)
    repo.set_status("d1", "failed", error="boom")
    assert repo.get("d1").error == "boom" and repo.get("d1").chunk_count == 7


def test_list_filters_by_access_level_and_status(repo):
    repo.upsert(_rec(1, access_level="internal"))
    repo.upsert(_rec(2, access_level="restricted"))
    repo.upsert(_rec(3, access_level="public", status="failed"))
    assert [d.document_id for d in repo.list()] == ["d1", "d2", "d3"]
    assert [d.document_id for d in repo.list(["public", "internal"])] == ["d1", "d3"]
    assert repo.list([]) == []
    assert [d.document_id for d in repo.list(status="failed")] == ["d3"]


def test_delete_cascades_ingestion_records(repo):
    repo.upsert(_rec(1))
    rid = repo.add_ingestion_record("d1", 1, "h1")
    repo.finish_ingestion_record(rid, "indexed", chunk_count=3)
    rec = repo.list_ingestion_records("d1")[0]
    assert rec.status == "indexed" and rec.chunk_count == 3 and rec.finished_at
    assert repo.delete("d1") is True and repo.delete("d1") is False
    assert repo.list_ingestion_records("d1") == []
