"""Dataset validity (grounded in the real synthetic KB) and deterministic text matching."""
from __future__ import annotations

import collections

import pytest

from copilot.config import PROJECT_ROOT
from copilot.evaluation.schemas import CATEGORIES, EvalCase, load_dataset, validate_dataset
from copilot.evaluation.text import contains, forbidden_hits, normalize

DATASET = PROJECT_ROOT / "data" / "eval" / "dataset.json"
KB = PROJECT_ROOT / "data" / "kb_documents"


def test_normalisation_and_word_boundaries():
    n = normalize("Error SL-401: after 5 failed attempts, locked for 15 minutes. vpn.acme-demo.example")
    assert contains(n, "sl 401") and contains(n, "SL-401") and contains(n, "vpn acme demo example")
    assert contains(n, "5") and not contains(n, "1") and not contains(n, "150")
    assert contains(n, "15 minutes") and not contains(n, "5 minutes")


def test_forbidden_hits():
    assert forbidden_hits("Use the paging-tool now", ["paging tool", "safe"]) == ["paging tool"]
    assert forbidden_hits("nothing here", ["paging tool"]) == []


def test_real_dataset_is_valid_and_grounded_in_the_knowledge_base():
    cases = load_dataset(DATASET)
    assert validate_dataset(cases, KB) == []
    assert len(cases) >= 40
    counts = collections.Counter(c.category for c in cases)
    assert set(counts) == set(CATEGORIES) and all(n >= 4 for n in counts.values())


def test_dataset_covers_roles_routes_and_access_control_cases():
    cases = load_dataset(DATASET)
    assert {c.expected_route for c in cases} == {"private_only", "private_then_external", "private_plus_external", "abstain", "out_of_scope"}
    assert any(c.role == "admin" for c in cases)
    leak_checks = [c for c in cases if c.forbidden_keywords]
    assert all(c.role == "employee" for c in leak_checks)
    assert len([c for c in leak_checks if c.expected_abstention]) >= 2  # restricted content must be refused
    assert any(not c.expected_abstention for c in leak_checks)  # answerable from a public document, restricted details must not leak
    assert all(c.expects_conflict_notice for c in cases if c.category == "conflicting")


def test_dataset_covers_multi_document_ambiguous_and_search_disabled_cases():
    cases = load_dataset(DATASET)
    assert any(len(c.expected_documents) > 1 and c.category != "conflicting" for c in cases)  # needs evidence from two documents
    assert any(not c.external_search_allowed for c in cases)  # search forbidden for the request
    assert any("ambiguous" in c.notes for c in cases)
    assert any("injection" in c.notes for c in cases)


def test_expected_document_ids_are_stable_hashes():
    c = next(c for c in load_dataset(DATASET) if c.id == "D01")
    assert c.expected_document_ids == ["e9dedc92d7347ec41418059ecb6e7c215401edbb55a23d5bb23cdfe4016d33f8"]


def test_validate_dataset_reports_problems(tmp_path):
    good = dict(id="X1", category="direct_kb", question="q", expected_route="private_only", expected_documents=["VPN_Connection_Guide.txt"],
                expected_answer_points=[{"text": "t", "keywords": ["nonexistentword"]}], expected_abstention=False)
    cases = [EvalCase(**good), EvalCase(**{**good, "expected_documents": ["Nope.txt"], "expected_answer_points": []})]
    problems = validate_dataset(cases, KB, min_cases=40)
    text = " | ".join(problems)
    assert "only 2 cases" in text and "missing categories" in text and "duplicate id X1" in text
    assert "unknown document Nope.txt" in text and "nonexistentword" in text


@pytest.mark.parametrize("bad", [
    dict(expected_route="abstain", expected_abstention=False),
    dict(expected_route="private_only", expected_abstention=False, expected_documents=[]),
    dict(expected_route="private_only", expected_abstention=False, expected_documents=["a.txt"], expected_citations=["b.txt"]),
])
def test_eval_case_rejects_inconsistent_labels(bad):
    with pytest.raises(ValueError):
        EvalCase(id="X", category="direct_kb", question="q", **bad)
