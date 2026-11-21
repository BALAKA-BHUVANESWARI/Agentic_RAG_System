"""Retrieval regression check: measuring, the baseline file and the comparison rules. No embedding model, no network."""
from __future__ import annotations

import copy
import json

import pytest

from copilot.evaluation import regression
from copilot.evaluation.schemas import load_dataset
from eval_helpers import VPN_DOC, write_dataset

KS = [1, 3, 5]


def ctx(**over) -> dict:
    base = {"embedding_model": "m", "chunk_size": 800, "chunk_overlap": 120, "retrieval_pool": 20, "ks": KS, "rerank_enabled": False,
            "max_chunks_per_doc": None, "kb_files": ["a.txt", "b.txt"]}
    return base | over


def scores(recall: float = 1.0, mrr: float = 1.0) -> dict:
    out = {"doc_mrr": mrr, "chunk_mrr": mrr}
    for k in KS:
        out |= {f"doc_recall@{k}": recall, f"doc_ndcg@{k}": recall, f"chunk_ndcg@{k}": recall}
    return out


def per_case(n: int = 20, **kw) -> dict:
    return {f"Q{i:02d}": scores(**kw) for i in range(n)}


def baseline_of(cases: dict | None = None, c: dict | None = None) -> dict:
    return regression.make_baseline(cases if cases is not None else per_case(), c or ctx(), created_at="2026-01-01T00:00:00Z")


# ------------------------------------------------------------------------------ the baseline document
def test_compared_keys_are_recall_mrr_and_ndcg_for_documents_and_chunks():
    keys = regression.compared_keys([1, 5])
    assert keys == ["doc_recall@1", "doc_recall@5", "doc_mrr", "doc_ndcg@1", "doc_ndcg@5", "chunk_mrr", "chunk_ndcg@1", "chunk_ndcg@5"]


def test_make_baseline_stores_rounded_scores_the_setup_and_means():
    b = regression.make_baseline({"Q2": {**scores(), "doc_mrr": 0.123456}, "Q1": scores(0.5)}, ctx(), created_at="t")
    assert b["version"] == regression.VERSION and b["n_cases"] == 2 and b["created_at"] == "t" and b["context"] == ctx()
    assert list(b["per_case"]) == ["Q1", "Q2"]                                      # sorted: a refresh gives a small, reviewable diff
    assert b["per_case"]["Q2"]["doc_mrr"] == 0.1235 and b["metrics"]["doc_recall@5"] == 0.75
    assert set(b["per_case"]["Q1"]) == set(regression.compared_keys(KS))             # only the compared metrics are stored


def test_missing_scores_are_stored_as_null_and_ignored_in_means():
    b = regression.make_baseline({"Q1": {**scores(), "chunk_mrr": None}, "Q2": scores(0.5, 0.5)}, ctx())
    assert b["per_case"]["Q1"]["chunk_mrr"] is None
    assert b["metrics"]["chunk_mrr"] == pytest.approx(0.5)                           # only Q2 contributes to the mean


def test_dumps_round_trips_and_is_stable(tmp_path):
    b = baseline_of()
    text = regression.dumps(b)
    assert text.endswith("\n") and json.loads(text) == b and regression.dumps(json.loads(text)) == text
    path = tmp_path / "b.json"
    path.write_text(text, encoding="utf-8")
    assert regression.load_baseline(path) == b


def test_load_baseline_explains_how_to_create_a_missing_or_broken_file(tmp_path):
    with pytest.raises(ValueError, match="--update-retrieval-baseline"):
        regression.load_baseline(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        regression.load_baseline(bad)
    bad.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected shape"):
        regression.load_baseline(bad)


# ------------------------------------------------------------------------------ comparing
def test_an_unchanged_measurement_passes():
    rep = regression.compare(baseline_of(), per_case(), ctx(), 0.02)
    assert rep["ok"] and rep["problems"] == [] and rep["n_compared"] == 20 and rep["lost"] == [] and rep["gained"] == []
    assert {r["status"] for r in rep["rows"]} == {"same"} and len(rep["rows"]) == len(regression.compared_keys(KS))


def test_a_drop_inside_the_tolerance_passes_and_a_drop_beyond_it_fails():
    now = per_case()
    now["Q00"] = scores(recall=0.0, mrr=0.0)                                         # 1 of 20 questions lost: -0.05 on every mean
    assert regression.compare(baseline_of(), now, ctx(), 0.10)["ok"]
    rep = regression.compare(baseline_of(), now, ctx(), 0.02)
    assert not rep["ok"] and any("doc_recall@5 fell from 1.000 to 0.950" in p for p in rep["problems"])
    assert rep["lost"] == ["Q00"] and {r["status"] for r in rep["rows"]} == {"regressed"}


def test_an_improvement_is_reported_but_never_fails_the_check():
    base = per_case(recall=0.5, mrr=0.5)
    rep = regression.compare(baseline_of(base), per_case(), ctx(), 0.02)
    assert rep["ok"] and {r["status"] for r in rep["rows"]} == {"improved"} and len(rep["gained"]) == 20


def test_questions_are_compared_only_where_both_sides_have_them():
    base = per_case(20)
    now = {i: s for i, s in per_case(20).items() if i != "Q00"} | {"NEW": scores(0.0, 0.0)}
    rep = regression.compare(baseline_of(base), now, ctx(), 0.02)
    assert rep["ok"] and rep["n_compared"] == 19 and rep["new_cases"] == ["NEW"] and rep["dropped_cases"] == ["Q00"]   # a new question cannot fail the old baseline
    assert "not in the baseline" in regression.format_report(rep) and "no longer in the dataset" in regression.format_report(rep)


def test_no_overlap_is_a_problem_not_a_pass():
    rep = regression.compare(baseline_of({"A": scores()}), {"B": scores()}, ctx(), 0.02)
    assert not rep["ok"] and "No question appears in both" in rep["problems"][0] and rep["rows"] == []


@pytest.mark.parametrize("change", [
    {"embedding_model": "other"}, {"chunk_size": 400}, {"chunk_overlap": 0}, {"retrieval_pool": 10}, {"ks": [1, 3]}, {"rerank_enabled": True},
    {"max_chunks_per_doc": 2}, {"kb_files": ["a.txt"]},
])
def test_a_changed_setup_refuses_to_compare_instead_of_guessing(change):
    rep = regression.compare(baseline_of(), per_case(), ctx(**change), 0.02)
    key = next(iter(change))
    assert not rep["ok"] and rep["rows"] == [] and any(f"Setup changed: {key}" in p and "refresh the baseline" in p for p in rep["problems"])


def test_an_unknown_baseline_version_is_refused():
    old = baseline_of() | {"version": 99}
    rep = regression.compare(old, per_case(), ctx(), 0.02)
    assert not rep["ok"] and "version 99" in rep["problems"][0]


def test_the_report_text_names_the_problem_and_the_questions_that_lost_a_hit():
    now = per_case()
    now["Q03"] = scores(0.0, 0.0)
    text = regression.format_report(regression.compare(baseline_of(), now, ctx(), 0.02))
    assert text.startswith("Retrieval regression check: FAILED") and "PROBLEM: doc_recall@5 fell" in text and "lost a document hit at the largest K: Q03" in text
    assert regression.format_report(regression.compare(baseline_of(), per_case(), ctx(), 0.02)).startswith("Retrieval regression check: OK")


def test_compare_does_not_modify_its_inputs():
    base, now, c = baseline_of(), per_case(), ctx()
    before = copy.deepcopy((base, now, c))
    regression.compare(base, now, c, 0.02)
    assert (base, now, c) == before


# ------------------------------------------------------------------------------ measuring and the setup snapshot
def test_measure_scores_only_questions_that_name_expected_documents(retriever, kb, tmp_path):
    cases = load_dataset(write_dataset(tmp_path / "ds.json"))
    out = regression.measure(cases, retriever, pool=20, ks=KS)
    assert list(out) == ["D1"]                                                      # U1 and E1 name no document
    assert set(regression.compared_keys(KS)) <= set(out["D1"]) and 0.0 <= out["D1"]["doc_recall@5"] <= 1.0
    assert VPN_DOC in {r.document_name for r in kb}


def test_context_records_what_makes_two_measurements_comparable(settings):
    c = regression.context(settings, ["b.txt", "a.txt"])
    assert c["kb_files"] == ["a.txt", "b.txt"] and c["ks"] == list(settings.eval_ks) and c["embedding_model"] == settings.embedding_model_name
    assert (c["chunk_size"], c["chunk_overlap"], c["rerank_enabled"]) == (settings.chunk_size, settings.chunk_overlap, settings.rerank_enabled)
