"""Retrieval-only diagnosis: WHY does the reranker lower chunk MRR? No Groq, no Tavily, no tokens.

For every labelled case with expected documents it retrieves ONE pool (reranker on, so every chunk has a
cosine score and a cross-encoder score) and ranks that pool four ways: by cosine (= reranker off), by the
cross-encoder (= reranker on), and by blends  a*cosine + (1-a)*cross-encoder  (both min-max scaled within
the question's pool). Then it:

  1. prints chunk MRR per ordering, overall and per category;
  2. for each case the reranker HURT, shows where the expected chunk moved, what the reranker put first
     instead, and how many of the question's content words each of the two chunks shares. This tests the
     hypothesis "the cross-encoder follows word overlap, the embedder follows meaning" directly;
  3. reports whether a blend would have helped. The blend weights are tried on the SAME cases they are
     scored on, so any gain is a hypothesis for a held-out test, not a result.

Usage (project root, venv active):
    python scripts/diagnose_rerank.py
    python scripts/diagnose_rerank.py --cases A02 R01 R07 --alphas 0.3 0.5 0.7
Output: printed report + data/eval/results/rerank_diag_<timestamp>/per_case.csv
"""
from __future__ import annotations

import argparse
import csv
import re
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

_STOP = {
    "the", "and", "for", "are", "but", "not", "you", "your", "our", "can", "how", "what", "when", "where", "who",
    "why", "does", "did", "has", "have", "had", "was", "were", "will", "with", "from", "that", "this", "these",
    "those", "into", "onto", "about", "just", "need", "keeps", "keep", "get", "got", "its", "his", "her", "their",
    "them", "they", "then", "than", "any", "all", "out", "off", "should", "would", "could", "okay", "never",
    "something", "somebody", "someone", "another",
}


def words(text: str) -> set[str]:
    """Lower-case content words (3+ letters, no stop words, no stemming)."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOP}


def overlap(question_words: set[str], chunk) -> float:
    """Share of the question's content words that also appear in the chunk (title, section and text)."""
    if not question_words:
        return 0.0
    return len(question_words & words(f"{chunk.title} {chunk.section or ''} {chunk.text}")) / len(question_words)


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.5] * len(values) if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in values]


def orderings(chunks: list, alphas: list[float]) -> dict[str, list]:
    """The same pool ranked by cosine, by cross-encoder, and by blends of the two."""
    cos, rr = minmax([c.score for c in chunks]), minmax([c.rerank_score for c in chunks])
    out = {"cosine": sorted(chunks, key=lambda c: c.score, reverse=True),
           "rerank": sorted(chunks, key=lambda c: c.rerank_score, reverse=True)}
    for a in alphas:
        mix = {id(c): a * x + (1 - a) * y for c, x, y in zip(chunks, cos, rr)}
        out[f"blend{a:g}"] = sorted(chunks, key=lambda c: mix[id(c)], reverse=True)
    return out


def as_item(c) -> RetrievedItem:
    return RetrievedItem(document_name=c.document_name, chunk_id=c.chunk_id, score=c.score, text=c.text)


def label(c) -> str:
    return f"{c.document_name} | {c.section or c.title or '-'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=None, help="only these case ids")
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.25, 0.5, 0.75],
                        help="cosine weight in the blends (0 = cross-encoder only, 1 = cosine only)")
    args = parser.parse_args()
    setup_logging(level="WARNING")

    base = get_settings()
    cases = [c for c in load_dataset(base.eval_dataset_path) if c.expected_documents]
    if args.cases:
        cases = [c for c in cases if c.id in set(args.cases)]
    on = base.model_copy(update={"rerank_enabled": True, "rerank_candidates": max(base.rerank_candidates, base.eval_retrieval_pool)})
    retriever = Retriever(on, embedder=get_embedding_service(), store=PineconeStore(base))
    print(f"{len(cases)} cases; pool={base.eval_retrieval_pool}; blend cosine weights={args.alphas}")

    names: list[str] = []
    rows: list[dict] = []
    detail: list[tuple] = []
    for case in cases:
        res = retriever.retrieve(case.question, case.role, top_k=base.eval_retrieval_pool, min_score=0.0)
        chunks = res.chunks
        if not res.reranked or any(c.rerank_score is None for c in chunks):
            print(f"  skip {case.id}: no cross-encoder scores (reranker did not run)")
            continue
        ords = orderings(chunks, args.alphas)
        names = list(ords)
        mrr = {n: score_retrieval(case, [as_item(c) for c in lst], base.eval_ks)["chunk_mrr"] for n, lst in ords.items()}
        rows.append({"case_id": case.id, "category": case.category, "question": case.question, **{f"mrr_{n}": mrr[n] for n in names}})
        if mrr["rerank"] < mrr["cosine"] - 1e-9:
            detail.append((case, ords, mrr))

    if not rows:
        print("Nothing scored.")
        return 1

    print(f"\n{'ordering':<12}{'overall':>9}" + "".join(f"{c:>18}" for c in sorted({r['category'] for r in rows})))
    cats = sorted({r["category"] for r in rows})
    for n in names:
        overall = sum(r[f"mrr_{n}"] for r in rows) / len(rows)
        per = []
        for cat in cats:
            sel = [r[f"mrr_{n}"] for r in rows if r["category"] == cat]
            per.append(f"{sum(sel) / len(sel):.3f} (n={len(sel)})")
        print(f"{n:<12}{overall:>9.3f}" + "".join(f"{p:>18}" for p in per))
    print("(chunk MRR; 'cosine' = reranker off, 'rerank' = reranker on)")

    print("\nHow each ordering compares with cosine, case by case:")
    for n in names[1:]:
        worse = sum(1 for r in rows if r[f"mrr_{n}"] < r["mrr_cosine"] - 1e-9)
        better = sum(1 for r in rows if r[f"mrr_{n}"] > r["mrr_cosine"] + 1e-9)
        print(f"  {n:<10} hurt {worse:>2}, helped {better:>2}, unchanged {len(rows) - worse - better:>2}")

    print(f"\n=== Cases the reranker HURT ({len(detail)}) ===")
    wording, other = 0, 0
    for case, ords, mrr in detail:
        q = words(case.question)
        rel = [c for c in ords["cosine"] if chunk_is_relevant(case, as_item(c))]
        cos_rank = {id(c): i + 1 for i, c in enumerate(ords["cosine"])}
        rr_rank = {id(c): i + 1 for i, c in enumerate(ords["rerank"])}
        print(f"\n{case.id} ({case.category})  chunk MRR {mrr['cosine']:.2f} -> {mrr['rerank']:.2f}\n  Q: {case.question}")
        if not rel:
            print("  (no chunk in the pool satisfies the answer-point keywords)")
            continue
        exp = min(rel, key=lambda c: rr_rank[id(c)])  # the relevant chunk the reranker ranks highest
        top = ords["rerank"][0]
        o_exp, o_top = overlap(q, exp), overlap(q, top)
        print(f"  expected chunk : {label(exp)}  cosine rank {cos_rank[id(exp)]} -> rerank rank {rr_rank[id(exp)]}"
              f"  (cosine {exp.score:.3f}, cross-encoder {exp.rerank_score:.4f}, shares {o_exp:.0%} of the question's words)")
        print(f"  reranker top-1 : {label(top)}  cosine rank {cos_rank[id(top)]}"
              f"  (cosine {top.score:.3f}, cross-encoder {top.rerank_score:.4f}, shares {o_top:.0%}; relevant: {'yes' if chunk_is_relevant(case, as_item(top)) else 'no'})")
        if o_top > o_exp:
            wording += 1
        else:
            other += 1
    if detail:
        print(f"\nWording check: in {wording} of {wording + other} hurt cases the chunk the reranker put first shares MORE of the "
              f"question's words than the expected chunk ({other} not). A high count supports 'the cross-encoder follows shared words'; "
              f"it does not prove it (no stemming, small sample, and shared words are only one thing a cross-encoder can use).")

    out_dir = base.eval_dir / "results" / time.strftime("rerank_diag_%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {out_dir}\nCaution: blend weights are tried on the same {len(rows)} cases they are scored on. A gain is a hypothesis for a "
          f"held-out set, not a result. Retrieval only; answer quality is not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
