"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from copilot.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test sees fresh settings."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def make_settings():
    """Build Settings ignoring any real .env file; override fields via kwargs."""

    def _make(**overrides):
        return Settings(_env_file=None, **overrides)

    return _make


# ---- Phase 2 fixtures: everything wired to fakes and a temp directory -------------
from copilot.db.repository import DocumentRepository  # noqa: E402
from copilot.db.schema import init_db  # noqa: E402
from copilot.ingestion.pipeline import IngestionPipeline  # noqa: E402
from copilot.vectorstore.pinecone_store import PineconeStore  # noqa: E402
from copilot.vectorstore.retriever import Retriever  # noqa: E402
from fakes import FakeEmbedder, FakeIndex  # noqa: E402


@pytest.fixture
def settings(make_settings, tmp_path):
    """Settings pointing at temp paths, no retry delay, small chunks, no score floor."""
    return make_settings(
        sqlite_path=tmp_path / "app.db",
        uploads_dir=tmp_path / "uploads",
        service_retry_base_delay_s=0.0,
        retrieval_min_score=0.0,
        chunk_size=300,
        chunk_overlap=40,
    )


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def index():
    return FakeIndex()


@pytest.fixture
def store(settings, index):
    return PineconeStore(settings, index=index)


@pytest.fixture
def repo(settings):
    init_db(settings.sqlite_path)
    return DocumentRepository(settings.sqlite_path)


@pytest.fixture
def pipeline(settings, embedder, store, repo):
    return IngestionPipeline(settings, embedder=embedder, store=store, repo=repo)


@pytest.fixture
def retriever(settings, embedder, store):
    return Retriever(settings, embedder=embedder, store=store)


# ---- Phase 3 fixtures ------------------------------------------------------------------
from copilot.baseline.rag import BaselineRAG  # noqa: E402
from copilot.generation.answer import AnswerGenerator  # noqa: E402


@pytest.fixture
def make_generator(settings):
    """Factory: AnswerGenerator wired to a FakeLLM with the given scripted outcomes."""
    from fakes import FakeLLM

    def _make(*outcomes):
        llm = FakeLLM(*outcomes)
        return AnswerGenerator(settings, llm=llm), llm

    return _make


@pytest.fixture
def make_baseline(settings, retriever):
    """Factory: BaselineRAG with real retriever (fake embedder/index) and a FakeLLM."""
    from fakes import FakeLLM

    def _make(*outcomes):
        llm = FakeLLM(*outcomes)
        return BaselineRAG(settings, retriever=retriever, generator=AnswerGenerator(settings, llm=llm)), llm

    return _make


# ---- Phase 4 fixtures: agent harness ------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from copilot.audit.logger import AuditLogger  # noqa: E402
from copilot.config import PROJECT_ROOT  # noqa: E402
from copilot.graph.deps import GraphDeps  # noqa: E402
from copilot.graph.runner import AgenticRAG  # noqa: E402


@pytest.fixture
def kb(pipeline):
    """Ingest the real synthetic KB (fake embedder, in-memory index)."""
    results = pipeline.ingest_directory(PROJECT_ROOT / "data" / "kb_documents")
    assert all(r.status == "indexed" for r in results)
    return results


@pytest.fixture
def make_agent(make_settings, tmp_path, embedder, index):
    """Factory: agent over the shared fake index with scripted LLM + fake search.

    Usage: h = make_agent(QueryClassification=[...], PrivateGrade=[...], search=FakeSearch(...), max_rewrite_retries=1)
    """
    from fakes import FakeSearch, ScriptedLLM

    def _make(search=None, retriever=None, settings_overrides=None, **scripts):
        s = make_settings(
            sqlite_path=tmp_path / "app.db", uploads_dir=tmp_path / "uploads", service_retry_base_delay_s=0.0,
            retrieval_min_score=0.0, chunk_size=300, chunk_overlap=40, **(settings_overrides or {}),
        )
        llm = ScriptedLLM(**scripts)
        search = search or FakeSearch()
        store = PineconeStore(s, index=index)
        retr = retriever or Retriever(s, embedder=embedder, store=store)
        audit = AuditLogger(s.sqlite_path)
        deps = GraphDeps(settings=s, llm=llm, retriever=retr, search=search, generator=AnswerGenerator(s, llm=llm), audit=audit)
        return SimpleNamespace(agent=AgenticRAG(s, deps), llm=llm, search=search, audit=audit, settings=s, store=store)

    return _make
