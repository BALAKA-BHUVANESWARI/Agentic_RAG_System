"""Unit tests for BaselineRAG: real ingestion + retrieval (fakes), scripted LLM."""
from __future__ import annotations

import pytest

from copilot.baseline.rag import BaselineRAG
from copilot.config import PROJECT_ROOT
from copilot.generation.answer import AnswerGenerator
from fakes import FakeLLM


@pytest.fixture
def kb(pipeline):
    pipeline.ingest_directory(PROJECT_ROOT / "data" / "kb_documents")


def test_baseline_retrieves_then_cites_the_first_chunk(kb, make_baseline):
    rag, llm = make_baseline({"can_answer": True, "answer": "Open the VPN client [1]."})
    resp = rag.answer("How do I connect to the company VPN from home?", "employee", request_id="req-1")
    assert resp.request_id == "req-1" and resp.mode == "baseline" and not resp.answer.abstained
    top = resp.retrieval.chunks[0]
    assert resp.answer.citations[0].chunk_id == top.chunk_id and resp.answer.citations[0].label == top.document_name
    assert top.text[:40] in llm.calls[0][1]["content"]
    assert resp.latency_ms > 0


def test_baseline_never_sends_restricted_text_to_the_llm_for_employees(kb, make_baseline):
    rag, llm = make_baseline({"can_answer": False})
    rag.answer("How does the service desk verify a caller before a password reset?", "employee")
    prompt = llm.calls[0][1]["content"]
    assert "Verify at least three items" not in prompt and "Service_Desk_Runbook" not in prompt


def test_unknown_role_gets_no_evidence_and_no_llm_call(kb, make_baseline):
    rag, llm = make_baseline()
    resp = rag.answer("How do I connect to the VPN?", "intruder")
    assert resp.answer.abstained and resp.answer.abstention_reason == "no_evidence" and llm.calls == []


def test_retrieval_outage_becomes_redacted_abstention(settings):
    class Boom:
        def retrieve(self, question, role):
            raise ConnectionError("pinecone down api_key=abc123456789")

    rag = BaselineRAG(settings, retriever=Boom(), generator=AnswerGenerator(settings, llm=FakeLLM()))
    resp = rag.answer("q", "employee")
    assert resp.answer.abstention_reason == "retrieval_unavailable" and resp.retrieval is None
    assert "abc123456789" not in (resp.error or "")


def test_blank_question_is_rejected(make_baseline):
    rag, _ = make_baseline()
    with pytest.raises(ValueError):
        rag.answer("   ")
