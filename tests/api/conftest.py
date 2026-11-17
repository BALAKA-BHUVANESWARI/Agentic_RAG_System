"""Fixtures for API tests: real routes/auth/ingestion/graph, fake LLM + search + in-memory index."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from copilot.api.main import create_app
from copilot.api.services import Services
from copilot.audit.feedback import FeedbackStore
from copilot.config import PROJECT_ROOT
from copilot.db.repository import DocumentRepository
from copilot.db.schema import init_db
from copilot.evaluation.runner import EvaluationRunner
from copilot.evaluation.service import EvaluationService
from copilot.evaluation.store import EvalRunStore
from copilot.ingestion.pipeline import IngestionPipeline
from copilot.vectorstore.retriever import Retriever

from eval_helpers import StubAgent, StubBaseline, write_dataset

from api_helpers import CLS_INTERNAL, DRAFT, GRADE_OK, OK_CHECK, SECRET_KEY


@pytest.fixture
def api(make_agent, embedder):
    """Factory: ``api(**llm_scripts, settings_overrides=...)`` -> SimpleNamespace(client, svc, h)."""

    def _build(settings_overrides=None, load_kb=True, **scripts):
        overrides = {"groq_api_key": SECRET_KEY, "pinecone_api_key": SECRET_KEY, "tavily_api_key": SECRET_KEY, **(settings_overrides or {})}
        h = make_agent(settings_overrides=overrides, **scripts)
        s = h.settings
        init_db(s.sqlite_path)
        repo = DocumentRepository(s.sqlite_path)
        pipeline = IngestionPipeline(s, embedder=embedder, store=h.store, repo=repo)
        if load_kb:
            assert all(r.status == "indexed" for r in pipeline.ingest_directory(PROJECT_ROOT / "data" / "kb_documents"))
        svc = Services(settings=s, agent=h.agent, pipeline=pipeline, repo=repo, audit=h.audit,
                       feedback=FeedbackStore(s.sqlite_path, s.max_comment_chars), store=h.store)
        return SimpleNamespace(client=TestClient(create_app(services=svc), raise_server_exceptions=False), svc=svc, h=h)

    return _build


@pytest.fixture
def chat_api(api):
    """API whose scripted LLM supports one full Scenario A answer."""
    return api(QueryClassification=[CLS_INTERNAL], PrivateGrade=[GRADE_OK], AnswerDraft=[DRAFT], GroundingCheck=[OK_CHECK])


@pytest.fixture
def eval_api(api, embedder, tmp_path):
    """Factory: API with a real EvaluationService over stub systems. Returns (ns, service)."""

    def _build(agent=None, baseline=None):
        ns = api()
        s = ns.h.settings.model_copy(update={"eval_dir": tmp_path / "eval", "eval_dataset_path": write_dataset(tmp_path / "ds.json")})
        runner = EvaluationRunner(
            s, agent=agent or StubAgent(), baseline=baseline or StubBaseline(), retriever=Retriever(s, embedder=embedder, store=ns.h.store),
            validate=False, sleep=lambda _s: None,
        )
        service = EvaluationService(runner, EvalRunStore(s.sqlite_path))
        ns.svc.evaluation_runner = service
        return ns, service

    return _build
