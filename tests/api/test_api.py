"""HTTP-level tests: contract, authorization, validation and error handling."""
from __future__ import annotations

import io

from api_helpers import ADM, CLS_INTERNAL, DRAFT, EMP, EMP2, GRADE_OK, OK_CHECK, SECRET_KEY

from copilot.ingestion.ids import make_document_id

RESTRICTED = {"Service_Desk_Runbook.txt", "Admin_Account_Recovery_Procedure.txt"}
NEW_DOC = b"SYNTHETIC DEMO DOCUMENT\nTitle: Printer Guide\n\n## 1. Printing\nUse the Follow-Me printer queue to print documents.\n"


def upload(client, headers, name="Printer_Guide.txt", data=NEW_DOC, **form):
    return client.post("/documents/upload", headers=headers, files={"file": (name, io.BytesIO(data), "text/plain")}, data=form)


# ---------------------------------------------------------------- health
def test_health_reports_dependencies_without_secrets(api):
    r = api(load_kb=False).client.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert {d["name"] for d in body["dependencies"]} == {"sqlite", "pinecone", "groq", "tavily"}
    assert SECRET_KEY not in r.text and "X-Request-ID" in r.headers


def test_health_degrades_when_keys_missing(api):
    r = api(load_kb=False, settings_overrides={"groq_api_key": None}).client.get("/health")
    groq = next(d for d in r.json()["dependencies"] if d["name"] == "groq")
    assert r.json()["status"] == "degraded" and not groq["ok"] and "GROQ_API_KEY" in groq["detail"]


# ---------------------------------------------------------------- chat
def test_chat_returns_answer_citations_and_real_trace(chat_api):
    r = chat_api.client.post("/chat", json={"question": "How do I connect to the company VPN from home?"}, headers=EMP)
    body = r.json()
    assert r.status_code == 200 and not body["abstained"] and body["citations"][0]["kind"] == "private"
    assert body["path"][0] == "query_classifier" and body["path"][-1] == "audit_logger"
    assert [e["node"] for e in body["trace"]] == body["path"]
    assert r.headers["X-Request-ID"] == body["request_id"] and body["tavily_calls"] == 0 and body["tokens"] > 0
    rec = chat_api.svc.audit.get(body["request_id"])
    assert rec.user_id == "alice" and rec.role == "employee"


def test_chat_without_headers_is_employee_anonymous(chat_api):
    body = chat_api.client.post("/chat", json={"question": "How do I connect to the VPN?"}).json()
    assert chat_api.svc.audit.get(body["request_id"]).role == "employee"
    assert chat_api.svc.audit.get(body["request_id"]).user_id == "anonymous"


def test_employee_chat_never_retrieves_restricted_documents(api):
    h = api(QueryClassification=[CLS_INTERNAL], PrivateGrade=[GRADE_OK], AnswerDraft=[DRAFT], GroundingCheck=[OK_CHECK],
            settings_overrides={"retrieval_top_k": 20})
    body = h.client.post("/chat", json={"question": "What are the service desk escalation and account recovery steps?"}, headers=EMP).json()
    restricted_ids = {make_document_id(n) for n in RESTRICTED}
    retrieved = {c.split(":")[0] for a in body["retrieval_attempts"] for c in a["chunk_ids"]}
    assert retrieved and not (retrieved & restricted_ids)


def test_chat_input_validation(chat_api):
    c = chat_api.client
    assert c.post("/chat", json={"question": "   "}, headers=EMP).status_code == 422
    assert c.post("/chat", json={}, headers=EMP).status_code == 422
    assert c.post("/chat", json={"question": "x" * 2001}, headers=EMP).status_code == 422
    assert c.post("/chat", json={"question": "hello", "session_id": "s" * 65}, headers=EMP).status_code == 422
    assert c.post("/chat", json={"question": "hello"}, headers={"X-Demo-Role": "superuser"}).status_code == 403
    assert c.post("/chat", json={"question": "hello"}, headers={"X-Demo-User": "bad user!"}).status_code == 400


def test_chat_external_flag_is_enforced(api):
    cls = dict(intent="external_current", needs_current_public_info=True, public_search_query="teams outage guidance")
    grade = dict(sufficient=False, confidence=0.1, recommended_action="external_search")
    h = api(settings_overrides={"max_rewrite_retries": 0}, QueryClassification=[cls], PrivateGrade=[grade])
    body = h.client.post("/chat", json={"question": "Latest Teams outage guidance?", "external_search_allowed": False}, headers=EMP).json()
    assert body["abstained"] and not body["external_search_triggered"] and h.h.search.call_count == 0


# ---------------------------------------------------------------- documents
def test_document_list_is_filtered_by_role_and_hides_admin_fields(api):
    h = api(load_kb=True)
    emp = h.client.get("/documents", headers=EMP).json()
    adm = h.client.get("/documents", headers=ADM).json()
    emp_names = {d["document_name"] for d in emp["documents"]}
    assert not (emp_names & RESTRICTED) and RESTRICTED <= {d["document_name"] for d in adm["documents"]}
    assert emp["count"] == len(emp["documents"]) < adm["count"]
    assert all(d["uploaded_by"] is None and d["error"] is None for d in emp["documents"])
    assert any(d["uploaded_by"] for d in adm["documents"])
    assert h.client.get("/documents?status=bogus", headers=ADM).status_code == 422
    assert all(d["status"] == "indexed" for d in h.client.get("/documents?status=indexed", headers=ADM).json()["documents"])


def test_upload_requires_admin_and_validates(api):
    h = api(load_kb=False, settings_overrides={"max_upload_mb": 1})
    c = h.client
    assert upload(c, EMP).status_code == 403
    ok = upload(c, ADM, category="printing", department="IT", access_level="restricted")
    assert ok.status_code == 201 and ok.json()["status"] == "indexed" and ok.json()["chunk_count"] >= 1
    assert upload(c, ADM, category="printing", department="IT", access_level="restricted").status_code == 200  # unchanged duplicate
    assert upload(c, ADM, name="malware.exe", data=b"MZ").status_code == 400
    assert upload(c, ADM, name="empty.txt", data=b"").status_code == 400
    assert upload(c, ADM, name="big.txt", data=b"a" * (1024 * 1024 + 1)).status_code == 413
    assert upload(c, ADM, name="x.txt", access_level="top-secret").status_code == 422
    doc = next(d for d in c.get("/documents", headers=ADM).json()["documents"] if d["document_name"] == "Printer_Guide.txt")
    assert doc["access_level"] == "restricted" and doc["uploaded_by"] == "root"
    assert "Printer_Guide.txt" not in {d["document_name"] for d in c.get("/documents", headers=EMP).json()["documents"]}


def test_delete_document(api):
    h = api(load_kb=False)
    doc_id = upload(h.client, ADM).json()["document_id"]
    assert h.client.delete(f"/documents/{doc_id}", headers=EMP).status_code == 403
    assert h.client.delete(f"/documents/{'0' * 64}", headers=ADM).status_code == 404
    assert h.client.delete("/documents/not-a-hash", headers=ADM).status_code == 422
    assert h.client.delete(f"/documents/{doc_id}", headers=ADM).json() == {"document_id": doc_id, "deleted": True}
    assert h.svc.repo.get(doc_id) is None and h.client.get("/documents", headers=ADM).json()["count"] == 0
    assert h.client.delete(f"/documents/{doc_id}", headers=ADM).status_code == 404


def test_reindex_document(api):
    h = api(load_kb=False)
    doc_id = upload(h.client, ADM).json()["document_id"]
    assert h.client.post(f"/documents/{doc_id}/reindex", headers=EMP).status_code == 403
    assert h.client.post(f"/documents/{'0' * 64}/reindex", headers=ADM).status_code == 404
    r = h.client.post(f"/documents/{doc_id}/reindex", headers=ADM)
    assert r.status_code == 200 and r.json()["status"] == "indexed"


# ---------------------------------------------------------------- feedback + audit
def test_feedback_only_on_own_requests(chat_api):
    c = chat_api.client
    rid = c.post("/chat", json={"question": "How do I connect to the VPN?"}, headers=EMP).json()["request_id"]
    ok = c.post("/feedback", json={"request_id": rid, "rating": "positive", "comment": "great, password=hunter2"}, headers=EMP)
    assert ok.status_code == 201 and ok.json()["rating"] == "positive"
    assert c.post("/feedback", json={"request_id": rid, "rating": "negative"}, headers=EMP2).status_code == 404
    assert c.post("/feedback", json={"request_id": rid, "rating": "negative"}, headers=ADM).status_code == 201
    assert c.post("/feedback", json={"request_id": "nope", "rating": "positive"}, headers=EMP).status_code == 404
    assert c.post("/feedback", json={"request_id": rid, "rating": "meh"}, headers=EMP).status_code == 422
    stored = chat_api.svc.feedback.for_request(rid)
    assert [f["rating"] for f in stored] == ["positive", "negative"] and "hunter2" not in stored[0]["comment"]


def test_audit_access_control(chat_api):
    c = chat_api.client
    rid = c.post("/chat", json={"question": "How do I connect to the VPN?"}, headers=EMP).json()["request_id"]
    c.post("/feedback", json={"request_id": rid, "rating": "positive"}, headers=EMP)
    own = c.get(f"/audit/{rid}", headers=EMP)
    assert own.status_code == 200 and own.json()["question"] and own.json()["trace"] and own.json()["feedback"][0]["rating"] == "positive"
    assert c.get(f"/audit/{rid}", headers=ADM).status_code == 200
    assert c.get(f"/audit/{rid}", headers=EMP2).status_code == 404  # existence not revealed
    assert c.get("/audit/doesnotexist", headers=ADM).status_code == 404
    assert c.get("/audit/bad!id", headers=ADM).status_code == 422


# ---------------------------------------------------------------- evaluation + errors
def test_evaluation_endpoint_is_admin_only_and_honest_until_phase_7(api):
    h = api(load_kb=False)
    assert h.client.post("/evaluation/run", json={}, headers=EMP).status_code == 403
    assert h.client.post("/evaluation/run", json={"mode": "bogus"}, headers=ADM).status_code == 422
    assert h.client.post("/evaluation/run", json={}, headers=ADM).status_code == 501


def test_unhandled_errors_do_not_leak_details(api):
    h = api(load_kb=False)

    class Boom:
        def answer(self, *a, **k):
            raise RuntimeError("internal secret detail api_key=abc123456789")

    h.svc.agent = Boom()
    r = h.client.post("/chat", json={"question": "hi there"}, headers=EMP)
    assert r.status_code == 500 and "secret detail" not in r.text and "abc123456789" not in r.text
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_openapi_documents_every_endpoint(api):
    paths = api(load_kb=False).client.get("/openapi.json").json()["paths"]
    for p in ("/chat", "/documents/upload", "/documents", "/documents/{document_id}", "/feedback", "/evaluation/run", "/audit/{request_id}", "/health"):
        assert p in paths


def test_upload_form_can_mark_a_document_superseded_and_the_list_shows_it(api):
    h = api(load_kb=False)
    assert upload(h.client, ADM, lifecycle="superseded", superseded_by="Printer_Guide_v2.txt").status_code == 201
    for headers in (ADM, EMP):  # the printer guide is internal, so employees may see its version status too
        docs = {d["document_name"]: d for d in h.client.get("/documents", headers=headers).json()["documents"]}
        assert (docs["Printer_Guide.txt"]["lifecycle"], docs["Printer_Guide.txt"]["superseded_by"]) == ("superseded", "Printer_Guide_v2.txt")
    assert upload(h.client, ADM, name="x.txt", lifecycle="obsolete").status_code == 422
    assert upload(h.client, EMP, name="y.txt", lifecycle="superseded").status_code == 403


def test_document_list_marks_the_legacy_password_policy_superseded_and_the_rest_current(api):
    h = api(load_kb=True)
    docs = {d["document_name"]: d for d in h.client.get("/documents", headers=EMP).json()["documents"]}
    legacy = docs.pop("Password_Policy_Legacy_2022.txt")
    assert (legacy["lifecycle"], legacy["superseded_by"]) == ("superseded", "Password_Policy.txt")
    assert {d["lifecycle"] for d in docs.values()} == {"current"}
