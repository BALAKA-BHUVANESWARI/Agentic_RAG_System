"""High-level entry point: run the agentic graph for one question."""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from copilot.audit.logger import AuditLogger
from copilot.auth.access import resolve_role
from copilot.config import Settings, get_settings
from copilot.generation.answer import AnswerGenerator, make_abstention
from copilot.generation.models import GeneratedAnswer
from copilot.graph.builder import build_graph
from copilot.graph.deps import GraphDeps
from copilot.graph.state import TraceEvent
from copilot.llm.client import LLMUsage
from copilot.logging_config import bind_request_id, get_logger, new_request_id, redact

log = get_logger(__name__)


class AgenticResponse(BaseModel):
    """Everything the API/UI/evaluation need from one agentic run."""

    request_id: str
    mode: str = "agentic"
    question: str
    role: str
    answer: GeneratedAnswer
    path: list[str]  # node names in the order they actually executed
    trace: list[TraceEvent]
    classification: dict | None = None
    private_grade: dict | None = None
    retrieval_attempts: list[dict] = Field(default_factory=list)
    rewritten_queries: list[str] = Field(default_factory=list)
    retry_count: int = 0
    external_search_triggered: bool = False
    external_query: str | None = None
    external_results: list[dict] = Field(default_factory=list)
    external_grade: dict | None = None
    external_blocked_reason: str | None = None
    validation: dict | None = None
    evidence: list[dict] = Field(default_factory=list)  # evidence items shown to the generator (used by the evaluation judge; never returned by /chat)
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    tavily_calls: int = 0
    latency_ms: float
    error: str | None = None


class AgenticRAG:
    """Builds dependencies and the graph once; ``answer`` runs one request."""

    def __init__(self, settings: Settings | None = None, deps: GraphDeps | None = None) -> None:
        self._settings = settings or get_settings()
        if deps is None:
            from copilot.llm.client import GroqClient
            from copilot.search.tavily_client import TavilySearchClient
            from copilot.vectorstore.retriever import Retriever

            llm = GroqClient(self._settings)
            deps = GraphDeps(
                settings=self._settings, llm=llm, retriever=Retriever(self._settings), search=TavilySearchClient(self._settings),
                generator=AnswerGenerator(self._settings, llm=llm), audit=AuditLogger(self._settings.sqlite_path),
            )
        self._deps = deps
        self._graph = build_graph(deps)

    def answer(
        self,
        question: str,
        role: str | None = None,
        *,
        user_id: str = "anonymous",
        session_id: str | None = None,
        external_search_allowed: bool = True,
        request_id: str | None = None,
    ) -> AgenticResponse:
        """Run the workflow. Failures become abstentions; the audit record is always attempted."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        rid = request_id or new_request_id()
        t0 = time.perf_counter()
        initial: dict[str, Any] = {
            "request_id": rid, "session_id": session_id or rid, "user_id": user_id,
            "role": resolve_role(role, self._settings), "question": question.strip(),
            "external_search_allowed": external_search_allowed, "started_perf": t0,
        }
        with bind_request_id(rid):
            try:
                final = self._graph.invoke(initial, {"recursion_limit": self._settings.graph_recursion_limit})
            except Exception as exc:  # noqa: BLE001 - includes GraphRecursionError
                error = redact(f"{type(exc).__name__}: {exc}")[:300]
                log.exception("Graph run failed")
                final = {**initial, "fatal_error": error, "trace": [],
                         "answer": make_abstention("internal_error", error=error, model=self._settings.groq_model).model_dump()}
                self._deps.audit.record_state(final, execution_time_ms=(time.perf_counter() - t0) * 1000)
        return self._to_response(final, (time.perf_counter() - t0) * 1000)

    def _to_response(self, final: dict[str, Any], latency_ms: float) -> AgenticResponse:
        trace = final.get("trace", [])
        answer = GeneratedAnswer(**final["answer"]) if final.get("answer") else make_abstention("internal_error")
        return AgenticResponse(
            request_id=final["request_id"], question=final["question"], role=final["role"], answer=answer,
            path=[e["node"] for e in trace], trace=trace,
            classification=final.get("classification"), private_grade=final.get("private_grade"),
            retrieval_attempts=final.get("retrieval_attempts", []), rewritten_queries=final.get("rewritten_queries", []),
            retry_count=final.get("retry_count", 0), external_search_triggered=bool(final.get("external_triggered")),
            external_query=final.get("external_query"), external_results=final.get("external_results", []),
            external_grade=final.get("external_grade"), external_blocked_reason=final.get("external_blocked_reason"),
            validation=final.get("validation"), evidence=final.get("evidence", []), llm_usage=LLMUsage(**(final.get("llm_usage") or {})),
            tavily_calls=final.get("tavily_calls", 0), latency_ms=latency_ms, error=final.get("fatal_error"),
        )
