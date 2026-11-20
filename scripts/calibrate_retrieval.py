"""Retrieval calibration (no Groq, no Tavily, no tokens): how many chunks to retrieve and which score cut-off to use.

Ranks a pool of chunks per dataset question with the reranker OFF (cosine order, ``min_score=0``) and reports:

1. TOP-K SWEEP: document recall@k and chunk precision@k for k = 3..10, overall and for multi-document questions
   (needs evidence from 2+ documents). Shows whether a larger ``RETRIEVAL_TOP_K`` brings in the missing document.
2. SCORE SEPARATION: cosine score of the best relevant chunk for answerable questions vs the top score of questions
   that have no private evidence. The score threshold (``RETRIEVAL_MIN_SCORE``) can only help if these two overlap little.
3. THRESHOLD SWEEP: for each cut-off, how many answerable questions would lose ALL their relevant chunks (too strict)
   and how many no-evidence questions would be stopped before grading (saved LLM calls).
4. PER-DOCUMENT CAP: real retrieval at RETRIEVAL_TOP_K with ``RETRIEVAL_MAX_CHUNKS_PER_DOC`` off, 3, 2 and 1: does a cap bring
   the missing second document into multi-document answers, and what does it cost in chunk ranking quality?

Usage (project root, venv active):
    python scripts/calibrate_retrieval.py
Output: printed tables + data/eval/results/calibration_<timestamp>/summary.json

Limits: small synthetic corpus and dataset; use the result to choose a sensible default and to see whether the current value is
clearly wrong, not to tune to single cases. The threshold is a cheap first filter; the LLM grader still decides sufficiency.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.embeddings.service import get_embedding_service  # noqa: E402
from copilot.evaluation.results import RetrievedItem  # noqa: E402
from copilot.evaluation.retrieval_metrics import chunk_is_relevant, score_retrieval  # noqa: E402
from copilot.evaluation.schemas import load_dataset  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402
from copilot.vectorstore.pinecone_store import PineconeStore  # noqa: E402
from copilot.vectorstore.retriever import Retriever  # noqa: E402

KS = [3, 4, 5, 6, 8, 10]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


def mean(values: list[float]) -> float | None:
    """Mean of a list, None when empty."""
    return sum(values) / len(values) if values else None


def pct(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100), None when empty."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))]


def fmt(v: float | None, spec: str = ".3f") -> str:
    """Format a possibly-missing number."""
    return "  n/a" if v is None else format(v, spec)


def main() -> int:
    setup_logging(level="WARNING")
    s = get_settings()
    off = s.model_copy(update={"rerank_enabled": False})
    embedder, store = get_embedding_service(), PineconeStore(s)
    retriever = Retriever(off, embedder=embedder, store=store)
    cases = load_dataset(s.eval_dataset_path)
    print(f"{len(cases)} cases; pool={s.eval_retrieval_pool}; current RETRIEVAL_TOP_K={s.retrieval_top_k}, RETRIEVAL_MIN_SCORE={s.retrieval_min_score}")

    ranked: dict[str, list[RetrievedItem]] = {}
    for c in cases:
        res = retriever.retrieve(c.question, c.role, top_k=s.eval_retrieval_pool, min_score=0.0)
        ranked[c.id] = [RetrievedItem(document_name=x.document_name, chunk_id=x.chunk_id, score=x.score, text=x.text) for x in res.chunks]

    answerable = [c for c in cases if c.expected_documents]
    multi = [c for c in answerable if len(c.expected_documents) > 1]
    no_evidence = [c for c in cases if not c.expected_documents and c.category != "out_of_scope"]
    out_of_scope = [c for c in cases if c.category == "out_of_scope"]

    # 1) top-k sweep
    scores = {c.id: score_retrieval(c, ranked[c.id], KS) for c in answerable}
    print(f"\n1) TOP-K SWEEP ({len(answerable)} answerable questions, {len(multi)} of them multi-document; cosine order, reranker off)")
    print(f"{'k':>3}  {'doc_recall':>10}  {'multi_doc_recall':>16}  {'chunk_precision':>15}")
    sweep = {}
    for k in KS:
        dr = mean([scores[c.id][f"doc_recall@{k}"] for c in answerable])
        mr = mean([scores[c.id][f"doc_recall@{k}"] for c in multi])
        cp = mean([scores[c.id][f"chunk_precision@{k}"] for c in answerable])
        sweep[k] = {"doc_recall": dr, "multi_doc_recall": mr, "chunk_precision": cp}
        print(f"{k:>3}  {fmt(dr):>10}  {fmt(mr):>16}  {fmt(cp):>15}")
    print("   multi-document questions still short of full recall at k=5:",
          [c.id for c in multi if scores[c.id]["doc_recall@5"] < 1] or "none")

    # 2) score separation
    best_rel: dict[str, float | None] = {}
    for c in answerable:
        rel = [i.score for i in ranked[c.id] if i.score is not None and chunk_is_relevant(c, i)]
        best_rel[c.id] = max(rel) if rel else None
    rel_scores = [v for v in best_rel.values() if v is not None]
    top1_noev = {c.id: (ranked[c.id][0].score if ranked[c.id] else 0.0) for c in no_evidence}
    top1_oos = {c.id: (ranked[c.id][0].score if ranked[c.id] else 0.0) for c in out_of_scope}
    print("\n2) SCORE SEPARATION (cosine similarity)")
    print(f"   best RELEVANT chunk, answerable questions (n={len(rel_scores)}): min {fmt(min(rel_scores) if rel_scores else None)}  "
          f"p10 {fmt(pct(rel_scores, 10))}  median {fmt(pct(rel_scores, 50))}")
    print(f"   TOP chunk, no-evidence questions (n={len(top1_noev)}): median {fmt(pct(list(top1_noev.values()), 50))}  "
          f"p90 {fmt(pct(list(top1_noev.values()), 90))}  max {fmt(max(top1_noev.values()) if top1_noev else None)}")
    print(f"   TOP chunk, out-of-scope questions (n={len(top1_oos)}; the graph declines these before retrieval): "
          f"max {fmt(max(top1_oos.values()) if top1_oos else None)}")
    print("   answerable questions with the lowest best-relevant score:",
          sorted(((round(v, 3), k) for k, v in best_rel.items() if v is not None))[:5])
    print("   answerable questions with NO relevant chunk in the pool:", [k for k, v in best_rel.items() if v is None] or "none")
    print("   no-evidence questions with the highest top score:", sorted(((round(v, 3), k) for k, v in top1_noev.items()), reverse=True)[:5])

    # 3) threshold sweep
    print("\n3) THRESHOLD SWEEP")
    print(f"{'min_score':>9}  {'answerable_lost':>15}  {'no_evidence_stopped':>19}  {'avg_chunks_kept@5':>17}")
    tsweep = {}
    for t in THRESHOLDS:
        lost = [k for k, v in best_rel.items() if v is None or v < t]
        stopped = [k for k, v in top1_noev.items() if v < t]
        kept = mean([float(len([i for i in ranked[c.id][:5] if (i.score or 0) >= t])) for c in answerable])
        tsweep[t] = {"answerable_lost": lost, "no_evidence_stopped": len(stopped), "avg_chunks_kept_at_5": kept}
        mark = "  <- current" if abs(t - s.retrieval_min_score) < 1e-9 else ""
        print(f"{t:>9.2f}  {len(lost):>9}/{len(answerable):<5}  {len(stopped):>13}/{len(no_evidence):<5}  {fmt(kept, '.2f'):>17}{mark}")
    safe = [t for t in THRESHOLDS if not tsweep[t]["answerable_lost"]]
    print(f"\n   highest cut-off that loses no answerable question: {max(safe) if safe else 'none'}"
          " (leave a margin below it; the LLM grader, not this threshold, decides sufficiency)")

    # 4) per-document cap at the real top-k
    k = s.retrieval_top_k
    print(f"\n4) PER-DOCUMENT CAP (real retrieval at top_k={k}, reranker off, min_score=0)")
    print(f"{'cap':>5}  {'doc_recall':>10}  {'multi_doc_recall':>16}  {'chunk_precision':>15}  {'chunk_mrr':>9}  multi-doc still short")
    cap_results = {}
    for cap in (None, 3, 2, 1):
        r = Retriever(off.model_copy(update={"retrieval_max_chunks_per_doc": cap}), embedder=embedder, store=store)
        per = {}
        for c in answerable:
            res = r.retrieve(c.question, c.role, top_k=k, min_score=0.0)
            items = [RetrievedItem(document_name=x.document_name, chunk_id=x.chunk_id, score=x.score, text=x.text) for x in res.chunks]
            per[c.id] = score_retrieval(c, items, [k])
        row = {
            "doc_recall": mean([per[c.id][f"doc_recall@{k}"] for c in answerable]),
            "multi_doc_recall": mean([per[c.id][f"doc_recall@{k}"] for c in multi]),
            "chunk_precision": mean([per[c.id][f"chunk_precision@{k}"] for c in answerable]),
            "chunk_mrr": mean([per[c.id]["chunk_mrr"] for c in answerable]),
            "multi_doc_short": [c.id for c in multi if per[c.id][f"doc_recall@{k}"] < 1],
        }
        cap_results["off" if cap is None else str(cap)] = row
        print(f"{'off' if cap is None else cap:>5}  {fmt(row['doc_recall']):>10}  {fmt(row['multi_doc_recall']):>16}  "
              f"{fmt(row['chunk_precision']):>15}  {fmt(row['chunk_mrr']):>9}  {row['multi_doc_short'] or 'none'}")
    print("   a cap only helps if multi-doc recall rises without chunk_mrr falling; the default stays off until answer quality is measured")

    out = s.eval_dir / "results" / time.strftime("calibration_%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "n_cases": len(cases), "current": {"top_k": s.retrieval_top_k, "min_score": s.retrieval_min_score},
        "top_k_sweep": sweep, "best_relevant_score": best_rel, "no_evidence_top1": top1_noev,
        "threshold_sweep": {str(t): v for t, v in tsweep.items()}, "per_document_cap": cap_results}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}\nNote: small synthetic corpus; retrieval only, answer quality not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
