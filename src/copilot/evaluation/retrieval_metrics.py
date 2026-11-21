"""Retrieval metrics: Recall@K, Precision@K, MRR and nDCG@K, at document level and chunk level.

Ground truth is the case's ``expected_documents``. A CHUNK counts as relevant when it comes from an
expected document AND contains the keywords of at least one expected answer point (so a relevant
chunk really carries answer evidence, without hand-labelling chunks).

Chunk-level recall is POOL-based: relevant chunks in the top-K divided by relevant chunks in the
deeper retrieved pool (``eval_retrieval_pool``). It estimates, but cannot guarantee, true recall.
"""
from __future__ import annotations

import math
from typing import Any

from copilot.evaluation.results import RetrievalCaseResult, RetrievedItem
from copilot.evaluation.schemas import EvalCase
from copilot.evaluation.text import all_keywords_present, normalize


def chunk_is_relevant(case: EvalCase, item: RetrievedItem) -> bool:
    """True if the chunk is from an expected document and satisfies at least one private answer point."""
    if item.document_name not in case.expected_documents:
        return False
    points = [p for p in case.expected_answer_points if p.source == "private"]
    if not points:
        return True
    text = normalize(item.text)
    return any(all_keywords_present(text, p.keywords) for p in points)


def _dcg(flags: list[bool]) -> float:
    return sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags) if f)


def ndcg_at_k(flags: list[bool], k: int, total_relevant: int) -> float | None:
    """Binary-relevance nDCG@K; ``total_relevant`` bounds the ideal ranking. None if nothing is relevant."""
    ideal = _dcg([True] * min(k, total_relevant))
    return _dcg(flags[:k]) / ideal if ideal else None


def reciprocal_rank(flags: list[bool]) -> float:
    """1/rank of the first relevant item (0.0 if none)."""
    for i, f in enumerate(flags, 1):
        if f:
            return 1.0 / i
    return 0.0


def score_retrieval(case: EvalCase, items: list[RetrievedItem], ks: list[int]) -> dict[str, float | None]:
    """All retrieval metrics for one case. Returns {} for cases with no expected documents."""
    if not case.expected_documents:
        return {}
    expected = set(case.expected_documents)
    chunk_flags = [chunk_is_relevant(case, i) for i in items]
    doc_flags = [i.document_name in expected for i in items]
    pool_relevant = sum(chunk_flags)
    out: dict[str, float | None] = {"doc_mrr": reciprocal_rank(doc_flags), "chunk_mrr": reciprocal_rank(chunk_flags)}
    for k in ks:
        top = items[:k]
        docs_top = {i.document_name for i in top}
        out[f"doc_recall@{k}"] = len(expected & docs_top) / len(expected)
        out[f"doc_precision@{k}"] = len(expected & docs_top) / len(docs_top) if docs_top else 0.0
        out[f"chunk_precision@{k}"] = sum(chunk_flags[:k]) / len(top) if top else 0.0
        out[f"chunk_recall@{k}"] = (sum(chunk_flags[:k]) / pool_relevant) if pool_relevant else 0.0
        nd = ndcg_at_k(chunk_flags, k, pool_relevant)
        out[f"chunk_ndcg@{k}"] = nd if nd is not None else 0.0
        dn = ndcg_at_k(doc_flags, k, sum(doc_flags))
        out[f"doc_ndcg@{k}"] = dn if dn is not None else 0.0
    return out


def rank_case(retriever: Any, case: EvalCase, pool: int, ks: list[int]) -> RetrievalCaseResult:
    """Rank a pool of chunks for the ORIGINAL question (no score floor, no LLM) and score it.

    The first K items are what a top-K retriever returns; query rewriting is not part of this measurement.
    Shared by the evaluation runner and the retrieval regression check so both measure the same thing.
    """
    res = retriever.retrieve(case.question, case.role, top_k=pool, min_score=0.0)
    items = [RetrievedItem(document_name=c.document_name, chunk_id=c.chunk_id, score=c.score, text=c.text) for c in res.chunks]
    return RetrievalCaseResult(case_id=case.id, category=case.category, items=items, scores=score_retrieval(case, items, ks))

