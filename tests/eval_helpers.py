"""Stubs and builders for evaluation tests (fakes live in tests/ only, never in app code)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

from copilot.baseline.rag import RAGResponse
from copilot.generation.citations import EXTERNAL_WARNING
from copilot.generation.models import Citation, EvidenceItem, GeneratedAnswer
from copilot.graph.runner import AgenticResponse
from copilot.llm.client import LLMUsage
from copilot.vectorstore.models import RetrievalResult, RetrievedChunk

VPN_Q = "How do I connect to the company VPN from home?"
UNANSWERABLE_Q = "What is the CEO's favourite colour?"
EXTERNAL_Q = "What is the latest Microsoft Teams outage guidance?"
VPN_DOC = "VPN_Connection_Guide.txt"
USAGE = LLMUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, calls=1)

DATASET = {
    "version": 1,
    "description": "tiny test dataset",
    "cases": [
        {"id": "D1", "category": "direct_kb", "question": VPN_Q, "expected_route": "private_only", "expected_documents": [VPN_DOC],
         "expected_answer_points": [{"text": "SecureLink client", "keywords": ["securelink"], "source": "private"}],
         "expected_citations": [VPN_DOC], "expected_abstention": False},
        {"id": "U1", "category": "unanswerable", "question": UNANSWERABLE_Q, "expected_route": "abstain", "expected_abstention": True},
        {"id": "E1", "category": "current_external", "question": EXTERNAL_Q, "expected_route": "private_then_external", "expected_abstention": False},
    ],
}


def write_dataset(path: Path, data: dict | None = None) -> Path:
    """Write the tiny dataset (or ``data``) and return the path."""
    path.write_text(json.dumps(data or DATASET), encoding="utf-8")
    return path


def make_answer(body: str = "Open the SecureLink client [1].", *, label: str = VPN_DOC, external: bool = False,
                url: str = "https://learn.microsoft.com/microsoftteams/") -> GeneratedAnswer:
    """A cited answer rendered the way the real generator renders it (warning, body, Sources)."""
    cit = Citation(ref=1, kind="external" if external else "private", label=label, location="document", url=url if external else None)
    parts = ([EXTERNAL_WARNING] if external else []) + [body, f"Sources:\n[1] {label} - {url if external else 'document'}"]
    return GeneratedAnswer(answer=body, rendered="\n\n".join(parts), citations=[cit], used_external=external, usage=USAGE)


def make_abstention(reason: str = "insufficient_evidence") -> GeneratedAnswer:
    """An abstention."""
    return GeneratedAnswer(rendered="I could not find reliable information. Please contact the IT Service Desk.",
                           abstained=True, abstention_reason=reason, usage=USAGE)  # type: ignore[arg-type]


def evidence_dicts(label: str = VPN_DOC) -> list[dict]:
    """One evidence item as stored on AgenticResponse."""
    return [EvidenceItem(ref=1, kind="private", label=label, location="document", text="Open the SecureLink client and connect.").model_dump()]


PATH_ANSWER = ["query_classifier", "private_retriever", "private_evidence_grader", "answer_generator", "final_answer_validator", "audit_logger"]
PATH_EXTERNAL = ["query_classifier", "private_retriever", "private_evidence_grader", "query_rewriter", "private_retriever", "private_evidence_grader",
                 "query_rewriter", "private_retriever", "private_evidence_grader", "external_search", "external_evidence_grader",
                 "answer_generator", "final_answer_validator", "audit_logger"]


def agentic_response(question: str, answer: GeneratedAnswer, **kw) -> AgenticResponse:
    """AgenticResponse with sensible defaults; override any field through ``kw``."""
    base = dict(request_id="req-1", question=question, role="employee", answer=answer, path=PATH_ANSWER, trace=[],
                private_grade={"effective_sufficient": True, "confidence": 0.9}, evidence=evidence_dicts(), validation={"passed": True},
                llm_usage=LLMUsage(prompt_tokens=300, completion_tokens=60, total_tokens=360, calls=3), latency_ms=1200.0)
    base.update(kw)
    return AgenticResponse(**base)


def baseline_response(question: str, answer: GeneratedAnswer) -> RAGResponse:
    """RAGResponse whose retrieval holds one chunk (the judge's evidence)."""
    chunk = RetrievedChunk(chunk_id="c1", document_id="d1", document_name=VPN_DOC, text="Open the SecureLink client and connect.", score=0.9)
    return RAGResponse(request_id="req-b", question=question, role="employee", answer=answer, latency_ms=500.0,
                       retrieval=RetrievalResult(query=question, role="employee", top_k=5, status="ok", chunks=[chunk]))


def default_agentic(question: str) -> AgenticResponse:
    """Correct agentic behaviour for the three tiny-dataset questions."""
    if question == VPN_Q:
        return agentic_response(question, make_answer())
    if question == UNANSWERABLE_Q:
        return agentic_response(question, make_abstention(), private_grade={"effective_sufficient": False, "confidence": 0.9}, evidence=[])
    return agentic_response(
        question, make_answer("Check the Microsoft 365 service health page [1].", label="Teams docs", external=True), path=PATH_EXTERNAL,
        private_grade={"effective_sufficient": False, "confidence": 0.8}, retry_count=2, rewritten_queries=["q one", "q two"],
        external_search_triggered=True, tavily_calls=1,
    )


def grade_rung_stub(question: str) -> AgenticResponse:
    """Stand-in for an ablation rung without web search: correct, except the public-information question is abstained."""
    if question == EXTERNAL_Q:
        return agentic_response(question, make_abstention("insufficient_evidence"), private_grade={"effective_sufficient": False, "confidence": 0.8}, evidence=[])
    return default_agentic(question)


def default_baseline(question: str) -> RAGResponse:
    """Baseline behaviour: answers the VPN question, hallucinates an answer to the unanswerable one, cannot answer the external one."""
    if question == VPN_Q:
        return baseline_response(question, make_answer())
    if question == UNANSWERABLE_Q:
        return baseline_response(question, make_answer("The favourite colour is blue [1]."))
    return baseline_response(question, make_abstention())


class StubAgent:
    """Agent double: ``responder(question)`` returns an AgenticResponse or an exception to raise. Records every call."""

    def __init__(self, responder: Callable = default_agentic, gate: threading.Event | None = None) -> None:
        self.responder, self.gate, self.calls = responder, gate, []

    def answer(self, question, role=None, *, user_id="x", session_id=None, external_search_allowed=True, request_id=None):
        self.calls.append({"question": question, "role": role, "user_id": user_id, "session_id": session_id, "external_allowed": external_search_allowed})
        if self.gate is not None:
            self.gate.wait(10)
        out = self.responder(question)
        if isinstance(out, BaseException):
            raise out
        return out


class StubBaseline:
    """Baseline double with the same contract."""

    def __init__(self, responder: Callable = default_baseline) -> None:
        self.responder, self.calls = responder, []

    def answer(self, question, role=None, request_id=None):
        self.calls.append(question)
        out = self.responder(question)
        if isinstance(out, BaseException):
            raise out
        return out
