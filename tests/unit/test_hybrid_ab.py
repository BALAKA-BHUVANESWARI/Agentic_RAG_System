"""Hybrid-retrieval experiment: corpus, access filtering, fusion wiring, summary statistics and the pre-registered decision rule.

Runs on the in-memory fake index and the fake embedder (no network, no model); the experiment itself is run with scripts/eval_hybrid_ab.py.
"""
from __future__ import annotations

import pytest

from copilot.evaluation import hybrid_ab
from copilot.evaluation.paired import MIN_CASES_FOR_VERDICT
from copilot.evaluation.retrieval_metrics import rank_case
from copilot.evaluation.schemas import EvalCase
from copilot.vectorstore.models import RetrievedChunk

VPN_DOC = "VPN_Connection_Guide.txt"
ADMIN_DOC = "Admin_Account_Recovery_Procedure.txt"
ALL_METRICS = [*hybrid_ab.DOC_METRICS, *hybrid_ab.CHUNK_METRICS]


def make_case(case_id: str, question: str, docs: list[str], *, role: str = "employee", category: str = "direct_kb") -> EvalCase:
    return EvalCase(id=case_id, category=category, question=question, role=role, expected_route="private_only", expected_documents=docs, expected_abstention=False)


@pytest.fixture
def corpus(kb, store):
    return hybrid_ab.LexicalCorpus(store.iter_chunks())


def names(items) -> set[str]:
    return {i.document_name for i in items}


# ------------------------------------------------------------------------------ the store's full listing and the corpus
def test_iter_chunks_returns_every_stored_chunk_with_text_and_metadata_and_is_not_access_filtered(kb, store, index, settings):
    chunks = list(store.iter_chunks(batch_size=3))                                   # several fetch batches
    assert {c.chunk_id for c in chunks} == set(index.ids(settings.pinecone_namespace)) and len(chunks) > 3
    assert all(c.text and c.document_name and c.access_level for c in chunks)
    assert {"restricted", "internal", "public"} <= {c.access_level for c in chunks}   # the caller filters, the store does not
    assert [c.chunk_id for c in store.iter_chunks(batch_size=1000)] == [c.chunk_id for c in chunks]


def test_iter_chunks_on_an_empty_store_is_empty(store):
    assert list(store.iter_chunks()) == []


def test_lexical_text_matches_what_is_embedded():
    chunk = RetrievedChunk(chunk_id="c", document_id="d", document_name="d.txt", title="VPN Guide", section="2. Connect", text="Open the client.")
    assert hybrid_ab.lexical_text(chunk) == "VPN Guide | 2. Connect\nOpen the client."
    assert hybrid_ab.lexical_text(chunk.model_copy(update={"section": None})) == "VPN Guide\nOpen the client."


def test_corpus_provenance_lists_documents_and_chunk_counts(corpus, store):
    info = corpus.describe()
    assert info["chunks"] == len(list(store.iter_chunks())) == sum(info["documents"].values())
    assert VPN_DOC in info["documents"] and ADMIN_DOC in info["documents"] and list(info["documents"]) == sorted(info["documents"])


# ------------------------------------------------------------------------------ the three arms
def test_the_dense_arm_is_exactly_the_applications_retrieval_for_the_original_question(kb, retriever, corpus, settings):
    case = make_case("Q1", "How do I connect to the company VPN from home?", [VPN_DOC])
    arms = hybrid_ab.rank_systems(case, retriever, corpus, settings, pool=10)
    direct = rank_case(retriever, case, 10, list(settings.eval_ks)).items
    assert [i.chunk_id for i in arms["dense"]] == [i.chunk_id for i in direct] and arms["dense"]


def test_bm25_finds_a_chunk_that_names_the_rare_term_in_the_question(kb, retriever, corpus, settings):
    case = make_case("Q1", "SecureLink VPN client", [VPN_DOC])
    arms = hybrid_ab.rank_systems(case, retriever, corpus, settings, pool=10)
    assert VPN_DOC in names(arms["bm25"][:5])
    scores = [i.score for i in arms["bm25"]]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)


def test_hybrid_only_contains_chunks_from_the_two_lists_without_duplicates_and_respects_the_pool(kb, retriever, corpus, settings):
    case = make_case("Q1", "reset my password and set up MFA", ["Password_Policy.txt"])
    arms = hybrid_ab.rank_systems(case, retriever, corpus, settings, pool=6)
    hybrid_ids = [i.chunk_id for i in arms["hybrid"]]
    assert len(hybrid_ids) == len(set(hybrid_ids)) <= 6
    assert set(hybrid_ids) <= {i.chunk_id for i in arms["dense"]} | {i.chunk_id for i in arms["bm25"]}
    scores = [i.score for i in arms["hybrid"]]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_is_just_dense_when_bm25_matches_nothing(kb, retriever, corpus, settings):
    case = make_case("Q1", "zzzzqqqq", [VPN_DOC])                                      # no chunk contains this token
    arms = hybrid_ab.rank_systems(case, retriever, corpus, settings, pool=5)
    assert arms["bm25"] == [] and [i.chunk_id for i in arms["hybrid"]] == [i.chunk_id for i in arms["dense"]]


# ------------------------------------------------------------------------------ access control
RESTRICTED_QUERY = "break-glass administrator account recovery procedure"


def test_an_employee_never_gets_a_restricted_chunk_in_any_arm(kb, retriever, corpus, settings, index):
    restricted_docs = {c.document_name for c in corpus.chunks.values() if c.access_level == "restricted"}
    assert ADMIN_DOC in restricted_docs
    case = make_case("Q1", RESTRICTED_QUERY, [ADMIN_DOC], role="employee")
    arms = hybrid_ab.rank_systems(case, retriever, corpus, settings, pool=20)
    for system, items in arms.items():
        assert not names(items) & restricted_docs, f"{system} returned a restricted document to an employee"
    assert arms["bm25"] and arms["dense"]                                              # the employee still gets (other) results


def test_an_admin_can_get_the_restricted_document_so_the_employee_test_is_not_vacuous(kb, retriever, corpus, settings):
    case = make_case("Q1", RESTRICTED_QUERY, [ADMIN_DOC], role="admin")
    arms = hybrid_ab.rank_systems(case, retriever, corpus, settings, pool=20)
    assert ADMIN_DOC in names(arms["bm25"]) and ADMIN_DOC in names(arms["hybrid"])


def test_an_unknown_role_gets_nothing_from_the_lexical_arm(kb, retriever, corpus, settings):
    case = make_case("Q1", "VPN", [VPN_DOC], role="contractor")
    assert hybrid_ab.rank_systems(case, retriever, corpus, settings, pool=10)["bm25"] == []


# ------------------------------------------------------------------------------ run_experiment
def test_run_experiment_scores_every_labelled_question_for_all_three_systems(kb, retriever, corpus, settings):
    labelled = make_case("Q1", "SecureLink VPN client", [VPN_DOC])
    abstain = EvalCase(id="U1", category="unanswerable", question="What is the CEO's favourite colour?", expected_route="abstain", expected_abstention=True)
    out = hybrid_ab.run_experiment([labelled, abstain], retriever, corpus, settings)
    assert set(out) == set(hybrid_ab.SYSTEMS)
    assert all(list(per_case) == ["Q1"] for per_case in out.values())                   # no expected documents -> not a retrieval question
    assert all(m in out[s]["Q1"] for s in out for m in hybrid_ab.DOC_METRICS)
    assert out["bm25"]["Q1"]["doc_recall@5"] == 1.0


# ------------------------------------------------------------------------------ summary statistics
def scores_for(value_of) -> dict[str, float | None]:
    return {m: value_of(m) for m in ALL_METRICS}


def make_per_system(n: int, dense, hybrid, bm25=None) -> dict:
    """``dense``/``hybrid``/``bm25``: callables ``(question_index, metric) -> score``."""
    ids = [f"Q{i:02d}" for i in range(n)]
    arms = {"dense": dense, "hybrid": hybrid, "bm25": bm25 or dense}
    return {s: {cid: scores_for(lambda m, i=i, f=f: f(i, m)) for i, cid in enumerate(ids)} for s, f in arms.items()}


def make_cases(n: int) -> list[EvalCase]:
    return [make_case(f"Q{i:02d}", "q", [VPN_DOC], category="direct_kb" if i % 2 else "paraphrased") for i in range(n)]


def summary(per_system: dict, n: int):
    ids = sorted(per_system["dense"])
    return hybrid_ab.summarize(per_system, make_cases(n), {"dev": set(ids[: n // 2]), "test": set(ids[n // 2 :])}, resamples=200, seed=1)


def test_a_clear_improvement_on_all_document_metrics_is_evidence_of_benefit():
    report = summary(make_per_system(20, lambda i, m: 0.5, lambda i, m: 0.8), 20)
    assert report["decision"]["outcome"] == "evidence_of_benefit" and set(report["decision"]["better"]) == set(hybrid_ab.DOC_METRICS)
    row = {r["metric"]: r for r in report["subsets"]["all"]["paired"]["hybrid"]}["doc_mrr"]
    assert row["verdict"] == "better" and row["n"] == 20 and row["improved"] == 20 and row["worsened"] == 0
    assert row["before"] == pytest.approx(0.5) and row["after"] == pytest.approx(0.8) and row["diff"] == pytest.approx(0.3)


def test_identical_arms_are_no_evidence_of_benefit_and_the_reason_says_that_is_not_proof_of_equality():
    report = summary(make_per_system(20, lambda i, m: 0.7, lambda i, m: 0.7), 20)
    assert report["decision"]["outcome"] == "no_evidence_of_benefit" and report["decision"]["better"] == [] and report["decision"]["worse"] == []
    assert "not evidence that the arms are equal" in report["decision"]["reason"]


def test_one_worse_document_metric_is_evidence_of_harm_even_when_others_improve():
    per_system = make_per_system(20, lambda i, m: 0.5, lambda i, m: 0.2 if m == "doc_recall@1" else 0.8)
    decision = summary(per_system, 20)["decision"]
    assert decision["outcome"] == "evidence_of_harm" and decision["worse"] == ["doc_recall@1"] and "doc_mrr" in decision["better"]


def test_a_single_better_metric_is_not_enough():
    per_system = make_per_system(20, lambda i, m: 0.5, lambda i, m: 0.9 if m == "doc_mrr" else 0.5)
    decision = summary(per_system, 20)["decision"]
    assert decision["outcome"] == "no_evidence_of_benefit" and decision["better"] == ["doc_mrr"]


def test_chunk_metrics_never_decide_the_outcome():
    per_system = make_per_system(20, lambda i, m: 0.5, lambda i, m: 0.9 if m in hybrid_ab.CHUNK_METRICS else 0.5)
    report = summary(per_system, 20)
    chunk_rows = [r for r in report["subsets"]["all"]["paired"]["hybrid"] if r["metric"] in hybrid_ab.CHUNK_METRICS]
    assert chunk_rows and all(r["verdict"] == "better" for r in chunk_rows)             # reported ...
    assert report["decision"]["outcome"] == "no_evidence_of_benefit"                    # ... but not decisive
    worse = make_per_system(20, lambda i, m: 0.5, lambda i, m: 0.1 if m in hybrid_ab.CHUNK_METRICS else 0.5)
    assert summary(worse, 20)["decision"]["outcome"] == "no_evidence_of_benefit"


def test_too_few_questions_give_no_verdict():
    n = MIN_CASES_FOR_VERDICT - 1
    report = summary(make_per_system(n, lambda i, m: 0.2, lambda i, m: 0.9), n)
    assert {r["verdict"] for r in report["subsets"]["all"]["paired"]["hybrid"]} == {"too few questions"}
    assert report["decision"]["outcome"] == "no_evidence_of_benefit" and "too few for a verdict" in report["decision"]["reason"]


def test_missing_scores_are_dropped_pairwise():
    base = make_per_system(12, lambda i, m: 0.5, lambda i, m: 0.9)
    base["dense"]["Q00"]["doc_ndcg@5"] = None
    rows = {r["metric"]: r for r in summary(base, 12)["subsets"]["all"]["paired"]["hybrid"]}
    assert rows["doc_ndcg@5"]["n"] == 11 and rows["doc_mrr"]["n"] == 12


def test_questions_that_changed_are_listed_by_document_mrr():
    def hybrid(i, m):
        return {0: 1.0, 1: 0.0}.get(i, 0.5) if m == "doc_mrr" else 0.5
    report = summary(make_per_system(12, lambda i, m: 0.5, hybrid), 12)
    assert report["changed_vs_dense_by_doc_mrr"] == {"improved": ["Q00"], "worsened": ["Q01"]}
    rows = {r["metric"]: r for r in report["subsets"]["all"]["paired"]["hybrid"]}
    assert rows["doc_mrr"]["improved"] == 1 and rows["doc_mrr"]["worsened"] == 1


def test_means_and_sizes_for_the_dev_and_test_subsets_and_the_categories():
    per_system = make_per_system(20, lambda i, m: 0.4 if i < 10 else 0.6, lambda i, m: 0.5, bm25=lambda i, m: 0.1)
    report = summary(per_system, 20)
    sub = report["subsets"]
    assert (sub["all"]["n"], sub["dev"]["n"], sub["test"]["n"]) == (20, 10, 10)
    assert sub["dev"]["means"]["dense"]["doc_mrr"] == pytest.approx(0.4) and sub["test"]["means"]["dense"]["doc_mrr"] == pytest.approx(0.6)
    assert sub["all"]["means"]["bm25"]["doc_mrr"] == pytest.approx(0.1) and {"bm25", "hybrid"} == set(sub["all"]["paired"])
    assert set(report["doc_mrr_by_category"]) == {"direct_kb", "paraphrased"} and report["doc_mrr_by_category"]["direct_kb"]["n"] == 10
    assert report["params"] == hybrid_ab.PARAMS and report["n_questions"] == 20 and report["caveats"]


# ------------------------------------------------------------------------------ the decision rule on its own
def rows_with(verdicts: dict[str, str], n: int = 30) -> list[dict]:
    return [{"metric": m, "verdict": verdicts.get(m, "not distinguishable"), "n": n} for m in ALL_METRICS]


@pytest.mark.parametrize("verdicts,outcome", [
    ({}, "no_evidence_of_benefit"),
    ({"doc_mrr": "better"}, "no_evidence_of_benefit"),
    ({"doc_mrr": "better", "doc_recall@1": "better"}, "evidence_of_benefit"),
    ({"doc_mrr": "better", "doc_recall@1": "better", "doc_ndcg@5": "worse"}, "evidence_of_harm"),
    ({"doc_recall@5": "worse"}, "evidence_of_harm"),
    ({"doc_mrr": "too few questions", "doc_recall@1": "too few questions"}, "no_evidence_of_benefit"),
    ({"chunk_mrr": "worse", "chunk_ndcg@5": "worse"}, "no_evidence_of_benefit"),
    ({"chunk_mrr": "better", "chunk_ndcg@5": "better"}, "no_evidence_of_benefit"),
])
def test_decision_rule_table(verdicts, outcome):
    result = hybrid_ab.decide(rows_with(verdicts))
    assert result["outcome"] == outcome and result["outcome"] in hybrid_ab.OUTCOMES and result["rule"] and result["reason"]
