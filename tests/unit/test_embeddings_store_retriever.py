"""Unit tests for EmbeddingService, PineconeStore and Retriever (fakes only, no network)."""
from __future__ import annotations

import pytest
from fakes import FakeIndex, make_chunk

from copilot.embeddings.service import EmbeddingService
from copilot.vectorstore.models import RetrievalFilters, RetrievedChunk
from copilot.vectorstore.pinecone_store import PineconeStore, chunk_to_metadata
from copilot.vectorstore.retriever import Retriever


# ---- EmbeddingService ----------------------------------------------------------
class _FakeModel:
    def __init__(self, dim: int) -> None:
        self.dim, self.calls = dim, []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[float(len(t))] * self.dim for t in texts]


def test_embedding_service_batches_and_normalizes_flag():
    model = _FakeModel(3)
    svc = EmbeddingService("m", 3, batch_size=8, model=model)
    out = svc.embed_documents(["ab", "abc"])
    assert out == [[2.0] * 3, [3.0] * 3]
    assert model.calls[0][1]["normalize_embeddings"] is True and model.calls[0][1]["batch_size"] == 8
    assert svc.embed_query("a") == [1.0] * 3


def test_embedding_service_empty_input_does_not_load_model():
    svc = EmbeddingService("never-loaded", 3, batch_size=8)
    assert svc.embed_documents([]) == []


def test_embedding_service_dimension_mismatch_raises():
    svc = EmbeddingService("m", 384, batch_size=8, model=_FakeModel(8))
    with pytest.raises(RuntimeError, match="384"):
        svc.embed_documents(["x"])


# ---- PineconeStore ---------------------------------------------------------------
def test_chunk_to_metadata_drops_none_values():
    md = chunk_to_metadata(make_chunk(page=None, section=None))
    assert "page_number" not in md and "section" not in md and "last_reviewed" not in md
    assert md["access_level"] == "internal" and md["chunk_id"] == "d1:v1:0"


def test_upsert_batches(make_settings, tmp_path):
    s = make_settings(pinecone_upsert_batch_size=2, sqlite_path=tmp_path / "x.db")
    idx = FakeIndex()
    store = PineconeStore(s, index=idx)
    chunks = [make_chunk(i) for i in range(5)]
    assert store.upsert(chunks, [[1.0, 0.0]] * 5) == 5
    assert idx.upsert_batches == [2, 2, 1]
    with pytest.raises(ValueError):
        store.upsert(chunks, [[1.0]])


def test_query_always_applies_access_filter(store, index):
    store.upsert([make_chunk(0, access="internal"), make_chunk(1, access="restricted", doc="d2")], [[1.0, 0.0], [1.0, 0.0]])
    res = store.query([1.0, 0.0], 5, ["public", "internal"])
    assert [r.access_level for r in res] == ["internal"]
    assert index.last_filter == {"access_level": {"$in": ["public", "internal"]}}


def test_query_combines_optional_filters_and_returns_models(store, index):
    store.upsert([make_chunk(0, page=3, section="S")], [[1.0, 0.0]])
    res = store.query([1.0, 0.0], 5, ["internal"], category="general", document_ids=["d1"])
    assert index.last_filter == {"$and": [
        {"access_level": {"$in": ["internal"]}}, {"category": {"$eq": "general"}}, {"document_id": {"$in": ["d1"]}},
    ]}
    assert isinstance(res[0], RetrievedChunk) and res[0].page_number == 3 and res[0].section == "S"
    assert res[0].location == "page 3, section 'S'"


def test_query_with_no_access_levels_does_not_hit_index(store, index):
    assert store.query([1.0], 5, []) == []
    assert index.query_calls == 0


def test_delete_document_all_and_keep_version(store, index):
    store.upsert(
        [make_chunk(0, version=1), make_chunk(1, version=1), make_chunk(0, version=2), make_chunk(0, doc="other")],
        [[1.0]] * 4,
    )
    assert store.delete_document("d1", keep_version=2) == 2  # listing is paginated (page_size=2)
    assert index.ids() == ["d1:v2:0", "other:v1:0"]
    assert store.delete_document("d1") == 1
    assert index.ids() == ["other:v1:0"]


def test_vector_count_and_health(store):
    assert store.vector_count() == 0
    store.upsert([make_chunk(0)], [[1.0]])
    assert store.vector_count() == 1 and store.health_check() is True


# ---- Retriever ---------------------------------------------------------------------
class _StubStore:
    """Returns fixed chunks regardless of filters, to prove the post-filter works."""

    def __init__(self, chunks):
        self.chunks, self.calls = chunks, 0

    def query(self, vector, top_k, access_levels, **kwargs):
        self.calls += 1
        return self.chunks


def _rc(chunk_id: str, level: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id=chunk_id, document_name=f"{chunk_id}.txt", access_level=level, score=score)


def test_retriever_post_filter_drops_unauthorized_even_if_store_leaks(settings, embedder):
    store = _StubStore([_rc("a", "internal", 0.9), _rc("b", "restricted", 0.95)])
    res = Retriever(settings, embedder=embedder, store=store).retrieve("vpn", "employee")
    assert [c.chunk_id for c in res.chunks] == ["a"] and res.dropped_unauthorized == 1 and res.status == "ok"


def test_retriever_admin_can_read_restricted(settings, embedder):
    store = _StubStore([_rc("b", "restricted", 0.95)])
    assert Retriever(settings, embedder=embedder, store=store).retrieve("vpn", "admin").chunks[0].chunk_id == "b"


def test_retriever_min_score_sorting_and_empty_status(settings, embedder):
    store = _StubStore([_rc("low", "internal", 0.1), _rc("mid", "internal", 0.5), _rc("high", "internal", 0.8)])
    r = Retriever(settings, embedder=embedder, store=store)
    res = r.retrieve("q", "employee", min_score=0.3)
    assert [c.chunk_id for c in res.chunks] == ["high", "mid"] and res.dropped_low_score == 1
    empty = r.retrieve("q", "employee", min_score=0.99)
    assert empty.status == "empty" and empty.chunks == []


def test_retriever_unknown_role_gets_no_access_without_querying(settings, embedder):
    store = _StubStore([_rc("a", "public", 0.9)])
    res = Retriever(settings, embedder=embedder, store=store).retrieve("q", "intruder")
    assert res.status == "no_access" and store.calls == 0


def test_retriever_rejects_empty_query_and_defaults_role(settings, embedder):
    r = Retriever(settings, embedder=embedder, store=_StubStore([]))
    with pytest.raises(ValueError):
        r.retrieve("   ")
    assert r.retrieve("q").role == "employee"


def test_retriever_passes_filters_and_top_k(settings, embedder, store, index):
    store.upsert([make_chunk(0)], [embedder.embed_query("vpn")])
    Retriever(settings, embedder=embedder, store=store).retrieve(
        "vpn", "employee", top_k=2, filters=RetrievalFilters(category="general")
    )
    assert index.last_filter == {"$and": [{"access_level": {"$in": ["public", "internal"]}}, {"category": {"$eq": "general"}}]}


def test_delete_document_accepts_sdk_list_items(store, index):
    """Current Pinecone SDKs list() yields pages of ListItem objects (with .id), not plain strings."""
    from types import SimpleNamespace

    store.upsert([make_chunk(0, version=1), make_chunk(0, version=2)], [[1.0]] * 2)
    original = index.list
    index.list = lambda prefix="", namespace=None: ([SimpleNamespace(id=i) for i in page] for page in original(prefix=prefix, namespace=namespace))
    assert store.delete_document("d1", keep_version=2) == 1
    assert index.ids() == ["d1:v2:0"]
