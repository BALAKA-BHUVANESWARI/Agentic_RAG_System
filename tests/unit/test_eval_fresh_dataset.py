"""The fresh held-out question block (data/eval/fresh_dataset.json, docs/FRESH_QUESTIONS.md).

It is separate from the 93-case dataset on purpose: it must never be tuned on, and the main dataset's size is pinned in
other tests. The fingerprint test fails if a label is edited without the edit being recorded.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from copilot.config import PROJECT_ROOT
from copilot.evaluation.schemas import CATEGORIES, load_dataset, validate_dataset

EVAL = PROJECT_ROOT / "data" / "eval"
FRESH = EVAL / "fresh_dataset.json"
FINGERPRINT = EVAL / "fresh_dataset.sha256"
KB = PROJECT_ROOT / "data" / "kb_documents"

pytestmark = pytest.mark.skipif(not FRESH.exists(), reason="the fresh block is not part of the Docker image; checked on the host")


def test_fresh_block_is_valid_and_grounded_in_the_knowledge_base():
    cases = load_dataset(FRESH)
    assert validate_dataset(cases, KB) == []  # at least 40 cases, every category, every label found in the cited documents
    assert len(cases) == 40 and {c.category for c in cases} == set(CATEGORIES)


def test_fresh_ids_and_questions_do_not_overlap_the_main_dataset():
    fresh, main = load_dataset(FRESH), load_dataset(EVAL / "dataset.json")
    assert {c.id for c in fresh}.isdisjoint({c.id for c in main})
    assert {c.question.strip().lower() for c in fresh}.isdisjoint({c.question.strip().lower() for c in main})
    assert all(c.id.startswith("F") for c in fresh)


def test_fresh_block_has_not_been_edited_since_it_was_frozen():
    """If this fails after a deliberate edit, regenerate the fingerprint AND record why in docs/FRESH_QUESTIONS.md."""
    cases = json.loads(FRESH.read_text(encoding="utf-8"))["cases"]
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert digest == FINGERPRINT.read_text(encoding="utf-8").strip()


def test_fresh_block_is_not_part_of_the_frozen_split():
    split = json.loads((EVAL / "split.json").read_text(encoding="utf-8"))
    assigned = set(split["dev"]) | set(split["test"])
    assert assigned.isdisjoint({c.id for c in load_dataset(FRESH)})
