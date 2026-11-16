"""CopilotClient against the real FastAPI app (fake LLM/search/index)."""
from __future__ import annotations

import pytest

from copilot.ui.api_client import ApiError, CopilotClient

NEW_DOC = b"SYNTHETIC DEMO DOCUMENT\nTitle: Printer Guide\n\n## 1. Printing\nUse the Follow-Me printer queue.\n"


def client_for(api_ns, role="employee", user="alice"):
    return CopilotClient("http://testserver", role, user, http=api_ns.client)


def test_chat_documents_feedback_audit_roundtrip(chat_api):
    c = client_for(chat_api)
    r = c.chat("How do I connect to the company VPN from home?")
    assert not r["abstained"] and r["citations"][0]["kind"] == "private"
    assert c.feedback(r["request_id"], "positive")["rating"] == "positive"
    assert c.audit(r["request_id"])["user_id"] == "alice"
    assert c.list_documents()["role"] == "employee" and c.health()["status"] == "ok"


def test_errors_are_readable(chat_api):
    c = client_for(chat_api)
    with pytest.raises(ApiError) as exc:
        c.chat("   ")
    assert exc.value.status_code == 422 and "question" in exc.value.detail
    with pytest.raises(ApiError) as exc:
        client_for(chat_api, role="superuser").chat("hello")
    assert exc.value.status_code == 403 and exc.value.detail == "Unknown role."
    with pytest.raises(ApiError) as exc:
        c.delete_document("0" * 64)
    assert exc.value.status_code == 403 and exc.value.request_id


def test_upload_failure_returns_body_but_success_and_admin_flow_work(api):
    ns = api(load_kb=False)
    admin = client_for(ns, "admin", "root")
    bad = admin.upload_document("malware.exe", b"MZ")
    assert bad["status"] == "failed"
    ok = admin.upload_document("Printer_Guide.txt", NEW_DOC, category="printing", access_level="internal")
    assert ok["status"] == "indexed" and admin.reindex_document(ok["document_id"])["status"] == "indexed"
    assert admin.delete_document(ok["document_id"])["deleted"] is True
    with pytest.raises(ApiError) as exc:
        admin.run_evaluation()
    assert exc.value.status_code == 501


def test_unreachable_backend_gives_a_helpful_message():
    c = CopilotClient("http://127.0.0.1:9", "employee", "alice", timeout=2)
    with pytest.raises(ApiError) as exc:
        c.health()
    assert exc.value.status_code == 0 and "run_api.py" in exc.value.detail
