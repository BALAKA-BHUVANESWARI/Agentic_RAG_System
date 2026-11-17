"""Application services container: built once at startup, injected into routes.

Routes depend on :func:`get_services`, so tests can provide a container with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from copilot.audit.feedback import FeedbackStore
from copilot.audit.logger import AuditLogger
from copilot.config import Settings
from copilot.db.repository import DocumentRepository
from copilot.db.schema import init_db
from copilot.ingestion.pipeline import IngestionPipeline


@dataclass
class Services:
    """Everything the API routes need."""

    settings: Settings
    agent: Any  # AgenticRAG-like: .answer(question, role, user_id=..., session_id=..., external_search_allowed=..., request_id=...)
    pipeline: IngestionPipeline
    repo: DocumentRepository
    audit: AuditLogger
    feedback: FeedbackStore
    store: Any  # needs .health_check()
    evaluation_runner: Any | None = None  # an EvaluationService (Phase 7b): start / status / list_runs / summary / cancel


def build_services(settings: Settings) -> Services:
    """Construct the real services (Groq, Pinecone, Tavily and the embedder are created lazily)."""
    from copilot.embeddings.service import get_embedding_service
    from copilot.generation.answer import AnswerGenerator
    from copilot.graph.deps import GraphDeps
    from copilot.graph.runner import AgenticRAG
    from copilot.llm.client import GroqClient
    from copilot.search.tavily_client import TavilySearchClient
    from copilot.vectorstore.pinecone_store import PineconeStore
    from copilot.vectorstore.retriever import Retriever

    init_db(settings.sqlite_path)
    embedder = get_embedding_service()
    store = PineconeStore(settings)
    repo = DocumentRepository(settings.sqlite_path)
    audit = AuditLogger(settings.sqlite_path)
    llm = GroqClient(settings)
    deps = GraphDeps(
        settings=settings, llm=llm, retriever=Retriever(settings, embedder=embedder, store=store),
        search=TavilySearchClient(settings), generator=AnswerGenerator(settings, llm=llm), audit=audit,
    )
    from copilot.evaluation.factory import build_evaluation_service

    agent = AgenticRAG(settings, deps)
    return Services(
        settings=settings, agent=agent, evaluation_runner=build_evaluation_service(settings, deps, agent),
        pipeline=IngestionPipeline(settings, embedder=embedder, store=store, repo=repo),
        repo=repo, audit=audit, feedback=FeedbackStore(settings.sqlite_path, settings.max_comment_chars), store=store,
    )


def get_services(request: Request) -> Services:
    """FastAPI dependency returning the container created at startup."""
    return request.app.state.services
