"""Retrieval-only A/B: reranker OFF vs ON over the labelled dataset. No Groq, no Tavily, no tokens.

Uses only Pinecone + local embeddings + the local cross-encoder. Both arms share the same embedder,
store, dataset and candidate pool (rerank_candidates == eval_retrieval_pool), so differences come
from the reranker alone. Mirrors EvaluationRunner._retrieval_pass (min_score=0, pool ranking).

Usage (project root, venv active):
    python scripts/eval_rerank_ab.py
    python scripts/eval_rerank_ab.py --categories direct_kb paraphrased needs_rewriting
Output: printed tables + data/eval/results/rerank_ab_<timestamp>/{summary.json,per_case.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.embeddings.service import get_embedding_service  # noqa: E402
from copilot.evaluation.aggregate import summarize_retrieval  # noqa: E402
from copilot.evaluation.results import RetrievalCaseResult, RetrievedItem  # noqa: E402
from copilot.evaluation.retrieval_metrics import score_retrieval  # noqa: E402
from copilot.evaluation.schemas import load_dataset  # noqa: E402
from copilot.evaluation.split import filter_cases  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402
from copilot.vectorstore.pinecone_store import PineconeStore  # noqa: E402
from copilot.vectorstore.retriever import Retriever  # noqa: E402

HEADLINE = ["doc_recall@5", "doc_mrr", "chunk_mrr", "chunk_precision@1", "chunk_precision@5", "chunk_ndcg@3", "chunk_ndcg@5"]


def run_arm(name: str, retriever: Retriever, cases, settings) -> dict[str, RetrievalCaseResult]:
    """Rank a pool per case and score it (same procedure as the eval runner)."""
    out: dict[str, RetrievalCaseResult] = {}
    for case in cases:
        res = retriever.retrieve(case.question, case.role, top_k=settings.eval_retrieval_pool, min_score=0.0)
        items = [RetrievedItem(document_name=c.document_name, chunk_id=c.chunk_id, score=c.score, text=c.text) for c in res.chunks]
        out[case.id] = RetrievalCaseResult(case_id=case.id, category=case.category, items=items,
                                           scores=score_retrieval(case, items, settings.eval_ks))
        print(f"  [{name}] {case.id} reranked={res.reranked} n={len(items)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="*", default=None, help="only these categories")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all", help="frozen split to use; tune on dev, not test")
    args = parser.parse_args()
    setup_logging(level="WARNING")

    base = get_settings()
    cases = [c for c in filter_cases(load_dataset(base.eval_dataset_path), args.split, base.eval_split_path) if c.expected_documents]
    if args.categories:
        cases = [c for c in cases if c.category in args.categories]
    print(f"{len(cases)} cases with expected documents; pool={base.eval_retrieval_pool}, rerank_candidates={base.rerank_candidates}")

    embedder, store = get_embedding_service(), PineconeStore(base)
    off_s = base.model_copy(update={"rerank_enabled": False})
    on_s = base.model_copy(update={"rerank_enabled": True, "rerank_candidates": max(base.rerank_candidates, base.eval_retrieval_pool)})
    arms = {
        "rerank_off": run_arm("off", Retriever(off_s, embedder=embedder, store=store), cases, off_s),
        "rerank_on": run_arm("on", Retriever(on_s, embedder=embedder, store=store), cases, on_s),
    }

    summaries = {n: summarize_retrieval(list(r.values()), base.eval_ks) for n, r in arms.items()}
    print(f"\n{'metric':<22}{'rerank_off':>12}{'rerank_on':>12}{'delta':>10}")
    for k in HEADLINE:
        a, b = (summaries[n]["metrics"].get(k) for n in ("rerank_off", "rerank_on"))
        if a is not None and b is not None:
            print(f"{k:<22}{a:>12.3f}{b:>12.3f}{b - a:>+10.3f}")

    print("\nPer-category chunk_mrr (off -> on):")
    for cat in sorted({c.category for c in cases}):
        ids = [c.id for c in cases if c.category == cat]
        m = [sum(arms[n][i].scores["chunk_mrr"] for i in ids) / len(ids) for n in ("rerank_off", "rerank_on")]
        print(f"  {cat:<22}n={len(ids):<3}{m[0]:.3f} -> {m[1]:.3f}")

    rows, worse, better = [], [], []
    for c in cases:
        off, on = arms["rerank_off"][c.id].scores, arms["rerank_on"][c.id].scores
        d = on["chunk_mrr"] - off["chunk_mrr"]
        (worse if d < -1e-9 else better if d > 1e-9 else []).append((c.id, c.question[:60], off["chunk_mrr"], on["chunk_mrr"]))
        rows.append({"case_id": c.id, "category": c.category, "question": c.question,
                     **{f"off_{k}": off.get(k) for k in HEADLINE}, **{f"on_{k}": on.get(k) for k in HEADLINE}})
    print(f"\nrerank HURT chunk_mrr on {len(worse)} case(s), HELPED on {len(better)}, unchanged on {len(cases) - len(worse) - len(better)}")
    for tag, lst in (("hurt", worse), ("helped", better)):
        for cid, q, a, b in lst:
            print(f"  {tag:<7}{cid}  {a:.2f} -> {b:.2f}  {q}")

    out_dir = base.eval_dir / "results" / time.strftime("rerank_ab_%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"n_cases": len(cases), "summaries": summaries}, indent=2), encoding="utf-8")
    with open(out_dir / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {out_dir}\nNote: n is small; treat deltas of a few points as noise. Retrieval only, answer quality not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
