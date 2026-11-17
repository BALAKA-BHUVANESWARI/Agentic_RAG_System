"""Evaluation dataset schema, loading and validation.

Documents are referenced by NAME in the JSON (readable); ``expected_document_ids`` derives the stable
document ids used by the system.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, Field, model_validator

from copilot.evaluation.text import all_keywords_present, normalize
from copilot.ingestion.ids import make_document_id

Category = Literal[
    "direct_kb", "paraphrased", "needs_rewriting", "insufficient_private",
    "current_external", "conflicting", "unanswerable", "out_of_scope",
]
Route = Literal["private_only", "private_then_external", "private_plus_external", "abstain", "out_of_scope"]
CATEGORIES: list[str] = list(get_args(Category))
_PRIVATE_ROUTES = {"private_only", "private_plus_external"}
_EXTERNAL_ROUTES = {"private_then_external", "private_plus_external"}


class AnswerPoint(BaseModel):
    """One fact an answer should contain. It is met when ALL keywords appear (word-boundary match)."""

    text: str
    keywords: list[str] = Field(min_length=1)
    source: Literal["private", "external"] = "private"


class EvalCase(BaseModel):
    """One labelled question."""

    id: str
    category: Category
    question: str
    role: str = "employee"
    expected_route: Route
    expected_documents: list[str] = Field(default_factory=list)
    expected_answer_points: list[AnswerPoint] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    expected_abstention: bool
    external_search_allowed: bool = True
    forbidden_keywords: list[str] = Field(default_factory=list)
    expects_conflict_notice: bool = False
    notes: str = ""

    @property
    def expected_document_ids(self) -> list[str]:
        """Stable ids of the expected documents."""
        return [make_document_id(n) for n in self.expected_documents]

    @property
    def expects_external(self) -> bool:
        """True if a public web search is part of the correct route."""
        return self.expected_route in _EXTERNAL_ROUTES

    @property
    def expects_sufficient_private_evidence(self) -> bool | None:
        """Ground truth for the evidence grader (None when grading is not meaningful, e.g. out of scope)."""
        if self.expected_route == "out_of_scope":
            return None
        return self.expected_route in _PRIVATE_ROUTES

    @model_validator(mode="after")
    def _consistent(self) -> "EvalCase":
        if self.expected_abstention != (self.expected_route in {"abstain", "out_of_scope"}):
            raise ValueError(f"{self.id}: expected_abstention contradicts expected_route '{self.expected_route}'")
        if self.expected_route in _PRIVATE_ROUTES and not self.expected_documents:
            raise ValueError(f"{self.id}: route '{self.expected_route}' needs expected_documents")
        if not set(self.expected_citations) <= set(self.expected_documents):
            raise ValueError(f"{self.id}: expected_citations must be a subset of expected_documents")
        return self


def load_dataset(path: Path | str) -> list[EvalCase]:
    """Load and validate the dataset file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase.model_validate(c) for c in raw["cases"]]


def validate_dataset(cases: list[EvalCase], kb_dir: Path | str, *, min_cases: int = 40) -> list[str]:
    """Return a list of problems (empty = valid): size, categories, duplicate ids and KB grounding of every label."""
    problems: list[str] = []
    if len(cases) < min_cases:
        problems.append(f"only {len(cases)} cases, need at least {min_cases}")
    ids = [c.id for c in cases]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    missing = set(CATEGORIES) - {c.category for c in cases}
    if missing:
        problems.append(f"missing categories: {sorted(missing)}")
    kb = {p.name: normalize(p.read_text(encoding="utf-8")) for p in Path(kb_dir).glob("*.txt")}
    all_text = " ".join(kb.values())
    for c in cases:
        for name in c.expected_documents:
            if name not in kb:
                problems.append(f"{c.id}: unknown document {name}")
        docs_text = " ".join(kb.get(n, "") for n in c.expected_documents)
        for p in c.expected_answer_points:
            if p.source == "private" and not all_keywords_present(docs_text, p.keywords):
                problems.append(f"{c.id}: point '{p.text}' keywords {p.keywords} not found in {c.expected_documents}")
        for f in c.forbidden_keywords:
            if not all_keywords_present(all_text, [f]):
                problems.append(f"{c.id}: forbidden keyword '{f}' appears in no document")
    return problems
