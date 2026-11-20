"""Pipeline tests: real pipeline + real SQLite + fake embedder + in-memory Pinecone index."""
from __future__ import annotations

import pytest
from fakes import make_docx_bytes, make_pdf_bytes

from copilot.config import PROJECT_ROOT
from copilot.ingestion.ids import make_document_id
from copilot.ingestion.pipeline import IngestionPipeline

DOC = (
    "SYNTHETIC DEMO DOCUMENT — NOT A REAL COMPANY POLICY\n"
    "Title: Test Guide\nCategory: vpn\nDepartment: IT\nAccess-Level: internal\n\n"
    "## Steps\n1. Open the client.\n2. Connect to the server.\n\n## Errors\nError 401 means bad password.\n"
)


def _ids(index):
    return index.ids()


def test_ingest_new_txt_indexes_and_tracks(pipeline, repo, index):
    r = pipeline.ingest_bytes("guide.txt", DOC.encode(), uploaded_by="alice")
    assert (r.status, r.version) == ("indexed", 1) and r.chunk_count >= 2
    rec = repo.get(make_document_id("guide.txt"))
    assert (rec.status, rec.version, rec.uploaded_by, rec.category) == ("indexed", 1, "alice", "vpn")
    assert rec.chunk_count == r.chunk_count and len(_ids(index)) == r.chunk_count
    assert all(i.startswith(f"{rec.document_id}:v1:") for i in _ids(index))
    assert [h.status for h in repo.list_ingestion_records(rec.document_id)] == ["indexed"]


def test_header_is_not_indexed_as_content(pipeline, index):
    pipeline.ingest_bytes("guide.txt", DOC.encode())
    texts = [md["text"] for _, md in index.data["kb"].values()]
    assert all("SYNTHETIC" not in t and "Access-Level" not in t for t in texts)
    assert {md["section"] for _, md in index.data["kb"].values()} == {"Steps", "Errors"}


def test_unchanged_reupload_is_skipped(pipeline, index):
    pipeline.ingest_bytes("guide.txt", DOC.encode())
    before = _ids(index)
    r = pipeline.ingest_bytes("guide.txt", DOC.encode())
    assert r.status == "skipped" and r.version == 1 and _ids(index) == before


def test_new_content_same_name_creates_new_version_and_removes_old_vectors(pipeline, repo, index):
    pipeline.ingest_bytes("guide.txt", DOC.encode())
    r = pipeline.ingest_bytes("guide.txt", (DOC + "\n## Extra\nNew section.\n").encode())
    assert (r.status, r.version) == ("indexed", 2)
    assert repo.get(r.document_id).version == 2
    assert _ids(index) and all(":v2:" in i for i in _ids(index))
    assert [h.version for h in repo.list_ingestion_records(r.document_id)] == [1, 2]


def test_duplicate_content_under_other_name_is_skipped(pipeline, index):
    pipeline.ingest_bytes("guide.txt", DOC.encode())
    count = len(_ids(index))
    r = pipeline.ingest_bytes("copy.txt", DOC.encode())
    assert r.status == "skipped" and "guide.txt" in r.message and len(_ids(index)) == count


@pytest.mark.parametrize("name,data", [("x.exe", b"MZ"), ("empty.txt", b""), ("bin.txt", b"a\x00b"), ("bad.pdf", b"nope")])
def test_malformed_files_fail_and_are_tracked(pipeline, repo, index, name, data):
    r = pipeline.ingest_bytes(name, data)
    assert r.status == "failed" and r.error and not _ids(index)
    assert repo.get(make_document_id(name)).status == "failed"
    assert repo.list_ingestion_records(make_document_id(name))[0].status == "failed"


def test_whitespace_only_document_fails(pipeline):
    r = pipeline.ingest_bytes("blank.txt", b"   \n\n  \t ")
    assert r.status == "failed" and "no extractable text" in r.message


def test_failed_reupload_does_not_disturb_indexed_document(pipeline, repo, index):
    pipeline.ingest_bytes("guide.txt", DOC.encode())
    before = _ids(index)
    r = pipeline.ingest_bytes("guide.txt", b"\x00\x01binary")
    assert r.status == "failed"
    assert repo.get(make_document_id("guide.txt")).status == "indexed" and _ids(index) == before


def test_access_level_override_and_validation(pipeline, repo):
    r = pipeline.ingest_bytes("guide.txt", DOC.encode(), access_level="restricted")
    assert repo.get(r.document_id).access_level == "restricted"
    bad = pipeline.ingest_bytes("other.txt", DOC.replace("internal", "x").encode(), access_level="secret")
    assert bad.status == "failed" and "access level" in bad.message


def test_docx_and_pdf_ingestion(pipeline, index):
    d = pipeline.ingest_bytes("g.docx", make_docx_bytes([("title", "G"), ("h1", "Overview"), ("p", "Use the VPN client.")]))
    p = pipeline.ingest_bytes("p.pdf", make_pdf_bytes(["Page one about VPN", "Page two about MFA"]))
    assert d.status == "indexed" and p.status == "indexed" and p.chunk_count == 2
    pages = sorted(md["page_number"] for _, md in index.data["kb"].values() if "page_number" in md)
    assert pages == [1, 2]


class _FailingStore:
    def upsert(self, chunks, vectors):
        raise ConnectionError("pinecone down api_key=abc123456789")

    def delete_document(self, *a, **k):
        return 0


def test_store_failure_marks_failed_without_leaking_secrets_then_retry_succeeds(settings, embedder, store, repo):
    bad = IngestionPipeline(settings, embedder=embedder, store=_FailingStore(), repo=repo)
    r = bad.ingest_bytes("guide.txt", DOC.encode())
    assert r.status == "failed" and "abc123456789" not in (r.error or "")
    assert repo.get(r.document_id).status == "failed"
    good = IngestionPipeline(settings, embedder=embedder, store=store, repo=repo)
    r2 = good.ingest_bytes("guide.txt", DOC.encode())  # failed docs are retried, same version
    assert (r2.status, r2.version) == ("indexed", 1)


def test_reindex_uses_stored_source_and_keeps_version(pipeline, repo, index):
    r = pipeline.ingest_bytes("guide.txt", DOC.encode(), access_level="restricted")
    index.data["kb"].clear()
    r2 = pipeline.reindex(r.document_id)
    assert (r2.status, r2.version) == ("indexed", 1) and len(_ids(index)) == r.chunk_count
    assert repo.get(r.document_id).access_level == "restricted"
    assert pipeline.reindex("nope").status == "failed"


def test_delete_removes_vectors_row_and_source(pipeline, repo, index, settings):
    r = pipeline.ingest_bytes("guide.txt", DOC.encode())
    assert pipeline.delete_document(r.document_id) is True
    assert not _ids(index) and repo.get(r.document_id) is None
    assert not list(settings.uploads_dir.glob("*"))
    assert pipeline.delete_document(r.document_id) is False


# ---- The real synthetic KB, end to end (fake embedder; no network) -----------------------
@pytest.fixture
def kb_loaded(pipeline):
    results = pipeline.ingest_directory(PROJECT_ROOT / "data" / "kb_documents")
    assert [r.message for r in results if r.status == "failed"] == []
    return results


def test_all_synthetic_documents_ingest(kb_loaded, repo):
    assert len(kb_loaded) == 11 and all(r.status == "indexed" for r in kb_loaded)
    levels = {d.document_name: d.access_level for d in repo.list()}
    assert levels["Service_Desk_Runbook.txt"] == "restricted" and levels["VPN_Connection_Guide.txt"] == "internal"
    assert levels["WiFi_and_Network_Access_Guide.txt"] == "public"


def test_vpn_question_finds_vpn_guide_with_section_citation(kb_loaded, retriever):
    res = retriever.retrieve("How do I connect to the company VPN from home?", "employee", top_k=5)
    top3 = res.chunks[:3]
    assert any(c.document_name == "VPN_Connection_Guide.txt" for c in top3)
    vpn = next(c for c in res.chunks if c.document_name == "VPN_Connection_Guide.txt")
    assert vpn.section and vpn.access_level == "internal" and vpn.version == 1 and vpn.last_reviewed == "2025-02-14"


def test_employee_never_sees_restricted_documents_but_admin_does(kb_loaded, retriever):
    q = "How does the service desk verify a caller before a password or MFA reset? ticket priorities escalation"
    employee = retriever.retrieve(q, "employee", top_k=10)
    assert all(c.access_level != "restricted" for c in employee.chunks)
    assert not any(c.document_name in {"Service_Desk_Runbook.txt", "Admin_Account_Recovery_Procedure.txt"} for c in employee.chunks)
    admin = retriever.retrieve(q, "admin", top_k=10)
    assert any(c.document_name == "Service_Desk_Runbook.txt" for c in admin.chunks[:3])


def test_conflicting_password_policies_are_both_retrievable(kb_loaded, retriever):
    res = retriever.retrieve("What is the minimum password length and how often must passwords be changed?", "employee", top_k=6)
    names = {c.document_name for c in res.chunks}
    assert {"Password_Policy.txt", "Password_Policy_Legacy_2022.txt"} <= names
    assert {c.last_reviewed for c in res.chunks if "Password_Policy" in c.document_name} >= {"2025-01-20", "2022-03-10"}


def test_teams_outage_question_has_no_strong_private_match(kb_loaded, retriever):
    """Scenario B precondition: nothing in the KB mentions outages, so no chunk text should."""
    res = retriever.retrieve("latest Microsoft Teams outage guidance", "admin", top_k=10)
    assert not any("outage" in c.text.lower() and "teams" in c.text.lower() for c in res.chunks)


def test_legacy_password_policy_is_indexed_as_superseded_and_current_one_is_not(kb_loaded, retriever):
    res = retriever.retrieve("What is the minimum password length and how often must passwords be changed?", "employee", top_k=6)
    old = [c for c in res.chunks if c.document_name == "Password_Policy_Legacy_2022.txt"]
    new = [c for c in res.chunks if c.document_name == "Password_Policy.txt"]
    assert old and new
    assert all((c.status, c.superseded_by) == ("superseded", "Password_Policy.txt") for c in old)
    assert all((c.status, c.superseded_by) == ("current", None) for c in new)
    assert all("Status:" not in c.text for c in old)  # the header line is metadata, not searchable text


def test_marking_a_document_superseded_is_a_new_version_not_a_skipped_duplicate(pipeline, index):
    pipeline.ingest_bytes("guide.txt", DOC.encode())
    assert {md["status"] for _, md in index.data["kb"].values()} == {"current"}
    superseded = DOC.replace("Access-Level: internal\n", "Access-Level: internal\nStatus: superseded by New_Guide.txt\n")
    r = pipeline.ingest_bytes("guide.txt", superseded.encode())
    assert (r.status, r.version) == ("indexed", 2)  # a header-only status change must not be skipped as unchanged
    metadata = [md for _, md in index.data["kb"].values()]
    assert metadata and all((md["status"], md["superseded_by"]) == ("superseded", "New_Guide.txt") for md in metadata)
    assert all(i.startswith(f"{make_document_id('guide.txt')}:v2:") for i in _ids(index))  # old vectors removed
    assert pipeline.ingest_bytes("guide.txt", superseded.encode()).status == "skipped"
