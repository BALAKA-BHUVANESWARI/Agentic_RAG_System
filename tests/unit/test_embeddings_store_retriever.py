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


# ---- Reranker + top-p (added 2026-11-19) --------------------------------------------
class _FakeCrossEncoder:
    """Scores a chunk by how many query words it contains (stands in for the real model)."""

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        if self.fail:
            raise RuntimeError("model unavailable")
        return [float(sum(w in text.lower() for w in query.lower().split())) for query, text in pairs]


def _tc(chunk_id: str, level: str, score: float, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id=chunk_id, document_name=f"{chunk_id}.txt", access_level=level, score=score, text=text)


def test_reranker_reorders_by_cross_encoder_and_keeps_top_k(settings, embedder):
    from copilot.vectorstore.reranker import CrossEncoderReranker

    store = _StubStore([
        _tc("a", "internal", 0.9, "parking rules"),
        _tc("b", "internal", 0.8, "vpn mfa approval steps"),
        _tc("c", "internal", 0.7, "vpn only"),
    ])
    r = Retriever(settings, embedder=embedder, store=store, reranker=CrossEncoderReranker("x", model=_FakeCrossEncoder()))
    res = r.retrieve("vpn mfa", "employee", top_k=2)
    assert res.reranked is True and [c.chunk_id for c in res.chunks] == ["b", "c"]
    assert res.chunks[0].rerank_score == 2.0 and res.chunks[0].score == 0.8  # cosine score is preserved


def test_reranker_never_sees_unauthorized_chunks(settings, embedder):
    from copilot.vectorstore.reranker import CrossEncoderReranker

    model = _FakeCrossEncoder()
    store = _StubStore([_tc("pub", "internal", 0.5, "vpn"), _tc("secret", "restricted", 0.9, "vpn vpn")])
    res = Retriever(settings, embedder=embedder, store=store, reranker=CrossEncoderReranker("x", model=model)).retrieve("vpn", "employee")
    assert [c.chunk_id for c in res.chunks] == ["pub"]
    assert all("secret" not in text for call in model.calls for _, text in call)


def test_reranker_failure_falls_back_to_vector_order(settings, embedder):
    from copilot.vectorstore.reranker import CrossEncoderReranker

    store = _StubStore([_tc("a", "internal", 0.9, "x"), _tc("b", "internal", 0.8, "y")])
    res = Retriever(settings, embedder=embedder, store=store, reranker=CrossEncoderReranker("x", model=_FakeCrossEncoder(fail=True))).retrieve("q", "employee")
    assert res.reranked is False and [c.chunk_id for c in res.chunks] == ["a", "b"]


def test_reranker_fetches_a_larger_candidate_pool(make_settings, tmp_path, embedder, store, index):
    from copilot.vectorstore.reranker import CrossEncoderReranker

    s = make_settings(sqlite_path=tmp_path / "a.db", retrieval_min_score=0.0, rerank_candidates=15)
    seen = {}
    real = store.query
    store.query = lambda vector, top_k, levels, **kw: seen.setdefault("k", top_k) and real(vector, top_k, levels, **kw)
    Retriever(s, embedder=embedder, store=store, reranker=CrossEncoderReranker("x", model=_FakeCrossEncoder())).retrieve("vpn", "employee", top_k=5)
    assert seen["k"] == 15


def test_rerank_text_adds_title_and_section_like_the_embedder():
    from copilot.vectorstore.reranker import rerank_text

    base = _tc("a", "internal", 0.9, "1. Open a P2 ticket.")
    assert rerank_text(base.model_copy(update={"title": "Admin Recovery", "section": "3. Recovering a locked account"})) == (
        "Admin Recovery | 3. Recovering a locked account\n1. Open a P2 ticket."
    )
    assert rerank_text(base.model_copy(update={"title": "Admin Recovery"})) == "Admin Recovery\n1. Open a P2 ticket."


def test_rerank_text_is_the_plain_text_without_title_or_section():
    from copilot.vectorstore.reranker import rerank_text

    assert rerank_text(_tc("a", "internal", 0.9, "vpn only")) == "vpn only"


def test_reranker_scores_the_heading_context_not_just_the_chunk_text():
    from copilot.vectorstore.reranker import CrossEncoderReranker

    model = _FakeCrossEncoder()
    chunk = _tc("a", "internal", 0.9, "1. Open a P2 ticket.").model_copy(update={"title": "Admin Recovery", "section": "Recovering a locked account"})
    other = _tc("b", "internal", 0.8, "parking rules")
    ordered, reranked = CrossEncoderReranker("x", model=model).rerank("locked account recovery", [other, chunk])
    assert reranked is True and [c.chunk_id for c in ordered] == ["a", "b"]  # only the heading contains "locked", "account"
    assert model.calls[0][1][1].startswith("Admin Recovery | Recovering a locked account\n")


# ---- Per-document chunk cap (added 2026-11-20) ---------------------------------------
def _dc(doc: str, n: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"{doc}:{n}", document_id=doc, document_name=f"{doc}.txt", access_level="internal", score=score)


class _RecordingStore(_StubStore):
    def query(self, vector, top_k, access_levels, **kwargs):
        self.last_top_k = top_k
        return super().query(vector, top_k, access_levels, **kwargs)


def test_chunk_cap_is_off_by_default(settings):
    assert settings.retrieval_max_chunks_per_doc is None


def test_chunk_cap_lets_a_second_document_into_the_top_k(make_settings, tmp_path, embedder):
    s = make_settings(sqlite_path=tmp_path / "a.db", retrieval_min_score=0.0, retrieval_max_chunks_per_doc=2)
    store = _StubStore([_dc("a", i, 0.9 - 0.05 * i) for i in range(5)] + [_dc("b", 0, 0.5), _dc("b", 1, 0.4)])
    res = Retriever(s, embedder=embedder, store=store).retrieve("q", "employee", top_k=5)
    assert [c.chunk_id for c in res.chunks] == ["a:0", "a:1", "a:2", "b:0", "b:1"]  # 2 + 2 within the cap, 1 refill; rank order kept


def test_chunk_cap_is_enforced_when_other_documents_are_available(make_settings, tmp_path, embedder):
    s = make_settings(sqlite_path=tmp_path / "a.db", retrieval_min_score=0.0, retrieval_max_chunks_per_doc=2)
    store = _StubStore([_dc("a", i, 0.9 - 0.05 * i) for i in range(5)] + [_dc("b", 0, 0.5), _dc("c", 0, 0.4), _dc("d", 0, 0.3)])
    res = Retriever(s, embedder=embedder, store=store).retrieve("q", "employee", top_k=5)
    assert [c.chunk_id for c in res.chunks] == ["a:0", "a:1", "b:0", "c:0", "d:0"]


def test_chunk_cap_never_returns_fewer_than_top_k_when_only_one_document_matches(make_settings, tmp_path, embedder):
    s = make_settings(sqlite_path=tmp_path / "a.db", retrieval_min_score=0.0, retrieval_max_chunks_per_doc=1)
    store = _StubStore([_dc("a", i, 0.9 - 0.05 * i) for i in range(4)])
    res = Retriever(s, embedder=embedder, store=store).retrieve("q", "employee", top_k=3)
    assert [c.chunk_id for c in res.chunks] == ["a:0", "a:1", "a:2"]


def test_chunk_cap_fetches_a_larger_candidate_pool_and_stays_off_otherwise(make_settings, tmp_path, embedder):
    on = make_settings(sqlite_path=tmp_path / "a.db", retrieval_max_chunks_per_doc=2, rerank_candidates=15)
    store = _RecordingStore([_dc("a", 0, 0.9)])
    Retriever(on, embedder=embedder, store=store).retrieve("q", "employee", top_k=5)
    assert store.last_top_k == 15
    off = make_settings(sqlite_path=tmp_path / "b.db", rerank_candidates=15)
    store = _RecordingStore([_dc("a", 0, 0.9)])
    Retriever(off, embedder=embedder, store=store).retrieve("q", "employee", top_k=5)
    assert store.last_top_k == 5


def test_chunk_cap_still_applies_access_control(make_settings, tmp_path, embedder):
    s = make_settings(sqlite_path=tmp_path / "a.db", retrieval_min_score=0.0, retrieval_max_chunks_per_doc=1)
    secret = RetrievedChunk(chunk_id="s:0", document_id="s", document_name="s.txt", access_level="restricted", score=0.99)
    store = _StubStore([secret, _dc("a", 0, 0.8), _dc("a", 1, 0.7), _dc("b", 0, 0.6)])
    res = Retriever(s, embedder=embedder, store=store).retrieve("q", "employee", top_k=3)
    assert [c.chunk_id for c in res.chunks] == ["a:0", "a:1", "b:0"]  # b gets a slot, one refill from a
    assert all(c.document_id != "s" for c in res.chunks)


def test_chunk_cap_setting_is_validated_and_blank_env_means_off(make_settings, monkeypatch):
    with pytest.raises(Exception):
        make_settings(retrieval_max_chunks_per_doc=0)
    monkeypatch.setenv("RETRIEVAL_MAX_CHUNKS_PER_DOC", "")
    assert make_settings().retrieval_max_chunks_per_doc is None
    monkeypatch.setenv("RETRIEVAL_MAX_CHUNKS_PER_DOC", "3")
    assert make_settings().retrieval_max_chunks_per_doc == 3


def test_retriever_without_reranker_is_unchanged_and_off_by_default(settings, embedder):
    assert settings.rerank_enabled is False
    store = _StubStore([_tc("a", "internal", 0.9, "x"), _tc("b", "internal", 0.8, "y")])
    res = Retriever(settings, embedder=embedder, store=store).retrieve("q", "employee")
    assert res.reranked is False and [c.chunk_id for c in res.chunks] == ["a", "b"]


def test_top_p_is_sent_only_when_configured(make_settings):
    from fakes import FakeGroqSDK
    from copilot.llm.client import GroqClient

    def kwargs_for(**overrides):
        sdk = FakeGroqSDK(["ok"])
        GroqClient(make_settings(groq_api_key="k", **overrides), client=sdk).chat([{"role": "user", "content": "hi"}])
        return sdk.calls[0]

    assert "top_p" not in kwargs_for()
    assert kwargs_for(groq_top_p=0.9)["top_p"] == 0.9


def test_top_p_is_validated_and_blank_env_means_default(make_settings, monkeypatch):
    with pytest.raises(Exception):
        make_settings(groq_top_p=0.0)
    with pytest.raises(Exception):
        make_settings(groq_top_p=1.5)
    monkeypatch.setenv("GROQ_TOP_P", "")
    from copilot.config import Settings

    assert Settings(_env_file=None).groq_top_p is None
