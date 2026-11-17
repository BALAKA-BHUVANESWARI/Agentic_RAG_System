"""Wiring of the real (Groq / Pinecone / Tavily) components for the evaluation runner, API and CLI."""
from __future__ import annotations

from copilot.baseline.rag import BaselineRAG
from copilot.config import Settings
from copilot.evaluation.judge import AnswerJudge
from copilot.evaluation.runner import EvaluationRunner
from copilot.evaluation.service import EvaluationService
from copilot.evaluation.store import EvalRunStore
from copilot.graph.deps import GraphDeps


def build_graph_deps(settings: Settings) -> GraphDeps:
    """Real dependencies for the agentic graph (clients are created lazily, so no network call happens here)."""
    from copilot.audit.logger import AuditLogger
    from copilot.generation.answer import AnswerGenerator
    from copilot.llm.client import GroqClient
    from copilot.search.tavily_client import TavilySearchClient
    from copilot.vectorstore.retriever import Retriever

    llm = GroqClient(settings)
    return GraphDeps(
        settings=settings, llm=llm, retriever=Retriever(settings), search=TavilySearchClient(settings),
        generator=AnswerGenerator(settings, llm=llm), audit=AuditLogger(settings.sqlite_path),
    )


def build_runner(settings: Settings, deps: GraphDeps, agent) -> EvaluationRunner:
    """Runner whose baseline and agent share the SAME retriever, generator and LLM client."""
    baseline = BaselineRAG(settings, retriever=deps.retriever, generator=deps.generator)
    return EvaluationRunner(
        settings, agent=agent, baseline=baseline, retriever=deps.retriever, judge=AnswerJudge(deps.llm, settings),
        store=EvalRunStore(settings.sqlite_path),
    )


def build_evaluation_service(settings: Settings, deps: GraphDeps, agent) -> EvaluationService:
    """Background-job service used by the API."""
    runner = build_runner(settings, deps, agent)
    return EvaluationService(runner, EvalRunStore(settings.sqlite_path))
