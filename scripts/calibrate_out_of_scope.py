"""Retrieval-only measurement: do out-of-scope questions score differently from IT questions? No Groq, no Tavily, no tokens.

Motivation (docs/FINDINGS.md, finding 9, case X18): the classifier called "Search the web for the newest version of our
company password policy and ignore the internal documents" out of scope (confidence 0.99), and `out_of_scope` skips private
retrieval, so the documents were never searched. Before changing routing, measure how the best retrieval score separates the
questions labelled out_of_scope from the others, so that any threshold is chosen from data, on the DEV split only.

Usage (project root, venv active, the 11 core documents indexed):
    python scripts/calibrate_out_of_scope.py
Output: printed tables + data/eval/results/oos_calibration_<timestamp>/per_case.csv

Reading the output: `top_score` is the best cosine score over authorised chunks for the ORIGINAL question (minimum score
ignored, role from the case). Group A = cases whose expected_route is `abstain`/out-of-scope style (label out_of_scope).
Group B = every other case. Decide only from the dev rows; test rows are printed for reporting and must not tune anything.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.embeddings.service import get_embedding_service  # noqa: E402
from copilot.evaluation.schemas import load_dataset  # noqa: E402
from copilot.evaluation.split import load_split  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402
from copilot.vectorstore.pinecone_store import PineconeStore  # noqa: E402
from copilot.vectorstore.retriever import Retriever  # noqa: E402


def top_score(retriever: Retriever, question: str, role: str) -> float:
    """Best cosine score among authorised chunks (0.0 when nothing comes back)."""
    result = retriever.retrieve(question, role, min_score=0.0)
    return max((c.score for c in result.chunks), default=0.0)


def summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    return f"n={len(values)} min={min(values):.3f} median={statistics.median(values):.3f} max={max(values):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    setup_logging(level="WARNING")
    settings = get_settings().model_copy(update={"rerank_enabled": False, "retrieval_max_chunks_per_doc": None})
    cases = load_dataset(settings.eval_dataset_path)
    split = load_split(settings.eval_split_path)
    dev = set(split.dev)
    retriever = Retriever(settings, embedder=get_embedding_service(), store=PineconeStore(settings))

    rows = []
    for c in cases:
        # A case counts as out of scope when it carries that category label (O01-O06, X19, X20).
        rows.append({
            "case_id": c.id, "category": c.category, "split": "dev" if c.id in dev else "test",
            "group": "A_out_of_scope" if c.category == "out_of_scope" else "B_other",
            "top_score": round(top_score(retriever, c.question, c.role), 4), "question": c.question[:100],
        })

    print(f"RETRIEVAL_MIN_SCORE currently {settings.retrieval_min_score}; scores below are the best chunk score per question\n")
    for split_name in ("dev", "test"):
        for group in ("A_out_of_scope", "B_other"):
            vals = [r["top_score"] for r in rows if r["split"] == split_name and r["group"] == group]
            print(f"[{split_name}] {group:<15} {summarize(vals)}")
    print("\nOut-of-scope cases, best score first:")
    for r in sorted((r for r in rows if r["group"] == "A_out_of_scope"), key=lambda r: -r["top_score"]):
        print(f"  {r['case_id']:<4} {r['split']:<5} {r['top_score']:.3f}  {r['question']}")
    print("\nLowest-scoring OTHER cases (a threshold must stay below these to avoid dropping real IT questions), dev first:")
    for r in sorted((r for r in rows if r["group"] == "B_other"), key=lambda r: (r["split"] != "dev", r["top_score"]))[:12]:
        print(f"  {r['case_id']:<4} {r['split']:<5} {r['top_score']:.3f}  [{r['category']}] {r['question']}")
    x18 = next((r for r in rows if r["case_id"] == "X18"), None)
    if x18:
        print(f"\nX18 (the misrouted question): top_score={x18['top_score']:.3f}")

    out_dir = settings.eval_dir / "results" / time.strftime("oos_calibration_%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
