"""LIVE tests (Groq + Pinecone + local MiniLM). Run: pytest tests/integration -m integration -s

Requires `python scripts/ingest_kb.py` to have been run once.
"""
from __future__ import annotations

import pytest

from copilot.baseline.rag import BaselineRAG
from copilot.llm.client import GroqClient

pytestmark = pytest.mark.integration


def test_groq_structured_output_roundtrip(need_groq):
    from pydantic import BaseModel

    class Pong(BaseModel):
        word: str

    parsed, usage = GroqClient().structured(
        [{"role": "user", "content": 'Return JSON with the key "word" set to "pong".'}], Pong
    )
    assert parsed.word.lower() == "pong" and usage.total_tokens > 0


def test_vpn_question_is_answered_from_the_vpn_guide(need_groq_and_pinecone):
    resp = BaselineRAG().answer("How do I connect to the company VPN from home?", "employee")
    print("\n" + resp.answer.rendered)
    assert not resp.answer.abstained, resp.answer.abstention_reason
    assert any(c.label == "VPN_Connection_Guide.txt" for c in resp.answer.citations)
    assert resp.answer.invalid_markers_removed == []


def test_employee_answer_never_cites_restricted_documents(need_groq_and_pinecone):
    resp = BaselineRAG().answer("How does the service desk verify a caller before a password reset?", "employee")
    print("\n" + resp.answer.rendered)
    restricted = {"Service_Desk_Runbook.txt", "Admin_Account_Recovery_Procedure.txt"}
    assert not any(c.label in restricted for c in resp.answer.citations)
    assert "Verify at least three items" not in resp.answer.rendered


def test_admin_can_get_answer_from_restricted_runbook(need_groq_and_pinecone):
    resp = BaselineRAG().answer("How does the service desk verify a caller before a password reset?", "admin")
    print("\n" + resp.answer.rendered)
    assert any(c.label == "Service_Desk_Runbook.txt" for c in resp.answer.citations)
