"""The fresh-2 held-out block (data/eval/fresh2_dataset.json, docs/FRESH2_QUESTIONS.md, docs/POST_DAY3_EXPERIMENTS.md).

Written before any experiment code or run. The fingerprint test fails if a label is edited without the edit being recorded.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from copilot.config import PROJECT_ROOT
from copilot.evaluation.schemas import CATEGORIES, load_dataset, validate_dataset

EVAL = PROJECT_ROOT / "data" / "eval"
FRESH2 = EVAL / "fresh2_dataset.json"
FINGERPRINT = EVAL / "fresh2_dataset.sha256"
KB = PROJECT_ROOT / "data" / "kb_documents"

pytestmark = pytest.mark.skipif(not FRESH2.exists(), reason="the fresh-2 block is not part of the Docker image; checked on the host")


def test_fresh2_block_is_valid_and_grounded_in_the_knowledge_base():
    cases = load_dataset(FRESH2)
    assert validate_dataset(cases, KB) == []
    assert len(cases) == 44 and {c.category for c in cases} == set(CATEGORIES)


def test_fresh2_has_the_planned_composition():
    cases = load_dataset(FRESH2)
    by_cat = {k: sum(c.category == k for c in cases) for k in CATEGORIES}
    assert by_cat == {
        "direct_kb": 6, "paraphrased": 8, "needs_rewriting": 6, "insufficient_private": 4,
        "current_external": 8, "conflicting": 2, "unanswerable": 8, "out_of_scope": 2,
    }
    external = [c for c in cases if c.category == "current_external"]
    assert sum(c.expected_route == "private_then_external" for c in external) == 4
    assert sum(c.expected_route == "private_plus_external" for c in external) == 4


def test_fresh2_does_not_overlap_any_other_block():
    fresh2 = load_dataset(FRESH2)
    others = load_dataset(EVAL / "dataset.json") + load_dataset(EVAL / "fresh_dataset.json")
    assert {c.id for c in fresh2}.isdisjoint({c.id for c in others})
    assert {c.question.strip().lower() for c in fresh2}.isdisjoint({c.question.strip().lower() for c in others})
    assert all(c.id.startswith("G") for c in fresh2)
    split = json.loads((EVAL / "split.json").read_text(encoding="utf-8"))
    assert (set(split["dev"]) | set(split["test"])).isdisjoint({c.id for c in fresh2})


def test_fresh2_block_has_not_been_edited_since_it_was_frozen():
    """If this fails after a deliberate edit, regenerate the fingerprint AND record why in docs/FRESH2_QUESTIONS.md."""
    cases = json.loads(FRESH2.read_text(encoding="utf-8"))["cases"]
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert digest == FINGERPRINT.read_text(encoding="utf-8").strip()
