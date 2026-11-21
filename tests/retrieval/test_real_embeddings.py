"""Retrieval regression check with the REAL embedding model (all-MiniLM-L6-v2) over the synthetic KB.

Skipped in a normal run because it loads the model (about 90 MB, cached after the first use). Run it with
``python -m pytest tests/retrieval --run-retrieval`` and refresh the committed baseline, deliberately, with
``python -m pytest tests/retrieval --update-retrieval-baseline``. No Pinecone, no LLM and no API key: an in-memory index.
"""
from __future__ import annotations

import pytest

from copilot.db.repository import DocumentRepository
from copilot.db.schema import init_db
from copilot.embeddings.service import build_embedding_service
from copilot.evaluation import regression
from copilot.evaluation.schemas import load_dataset
from copilot.ingestion.pipeline import IngestionPipeline
from copilot.vectorstore.pinecone_store import PineconeStore
from copilot.vectorstore.retriever import Retriever
from fakes import FakeIndex

pytestmark = pytest.mark.retrieval


def test_retrieval_has_not_regressed(make_settings, tmp_path, request):
    s = make_settings(sqlite_path=tmp_path / "app.db", uploads_dir=tmp_path / "uploads", retrieval_min_score=0.0, service_retry_base_delay_s=0.0)
    init_db(s.sqlite_path)
    embedder = build_embedding_service(s)                                    # the real model
    store = PineconeStore(s, index=FakeIndex(page_size=1000))              # in-memory; embeddings are normalised, so dot product = cosine
    pipeline = IngestionPipeline(s, embedder=embedder, store=store, repo=DocumentRepository(s.sqlite_path))
    results = pipeline.ingest_directory(s.kb_documents_dir)
    assert results and all(r.status == "indexed" for r in results), [(r.document_name, r.status, r.error) for r in results]

    cases = load_dataset(s.eval_dataset_path)
    per_case = regression.measure(cases, Retriever(s, embedder=embedder, store=store), pool=s.eval_retrieval_pool, ks=s.eval_ks)
    assert per_case, "no question in the dataset names expected documents"
    kb_files = [p.name for p in s.kb_documents_dir.iterdir() if p.is_file() and p.suffix.lower() in s.allowed_extensions]
    ctx = regression.context(s, kb_files)

    path = s.eval_retrieval_baseline_path
    if request.config.getoption("--update-retrieval-baseline"):
        path.write_text(regression.dumps(regression.make_baseline(per_case, ctx)), encoding="utf-8")
        pytest.skip(f"baseline written to {path} ({len(per_case)} questions); nothing was compared. Review the diff before committing it.")

    report = regression.compare(regression.load_baseline(path), per_case, ctx, s.eval_regression_tolerance)
    assert report["ok"], "\n" + regression.format_report(report)
