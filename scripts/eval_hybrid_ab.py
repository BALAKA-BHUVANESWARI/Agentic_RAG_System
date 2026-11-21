"""Retrieval-only experiment: dense vs BM25 vs hybrid (dense + BM25 fused with RRF). No Groq, no Tavily, no tokens.

Uses the existing Pinecone index (read-only: list + fetch + query) and the local embedding model. The design, the fixed
parameters and the decision rule are in docs/HYBRID_RETRIEVAL.md and were written before the first run. The reranker and the
per-document chunk cap are forced off so the dense arm is plain dense retrieval, comparable with the committed retrieval baseline.

Usage (project root, venv active, KB ingested with scripts/ingest_kb.py):
    python scripts/eval_hybrid_ab.py --corpus-only    # list what the lexical index would be built from, then stop
    python scripts/eval_hybrid_ab.py
Output: printed tables + data/eval/results/hybrid_ab_<timestamp>/{summary.json,per_case.csv}
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
from copilot.evaluation import hybrid_ab, regression  # noqa: E402
from copilot.evaluation.schemas import load_dataset  # noqa: E402
from copilot.evaluation.split import load_split  # noqa: E402
from copilot.evaluation.system import file_sha256  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402
from copilot.vectorstore.pinecone_store import PineconeStore  # noqa: E402
from copilot.vectorstore.retriever import Retriever  # noqa: E402


def _print_means(report: dict, subset: str) -> None:
    block = report["subsets"][subset]
    print(f"\n[{subset}] n={block['n']}  (means; document metrics decide the outcome)")
    print(f"{'metric':<16}" + "".join(f"{hybrid_ab.LABELS[s][:22]:>24}" for s in hybrid_ab.SYSTEMS))
    for m in [*hybrid_ab.DOC_METRICS, *hybrid_ab.CHUNK_METRICS]:
        flag = "  (chunk: keyword-defined relevance, not decisive)" if m in hybrid_ab.CHUNK_METRICS else ""
        cells = [block["means"][s][m] for s in hybrid_ab.SYSTEMS]
        print(f"{m:<16}" + "".join(f"{'n/a' if c is None else f'{c:.3f}':>24}" for c in cells) + flag)


def _print_paired(report: dict, subset: str, after: str) -> None:
    rows = report["subsets"][subset]["paired"][after]
    print(f"\n[{subset}] dense -> {after}  (paired bootstrap, {report['confidence']:.0%} interval; verdict only if the interval excludes 0 and n >= 10)")
    print(f"{'metric':<16}{'n':>4}{'dense':>8}{after:>8}{'diff':>8}{'interval':>20}{'better':>8}{'worse':>7}  verdict")
    for r in rows:
        interval = "[{:+.3f}, {:+.3f}]".format(r["lo"], r["hi"])
        print(f"{r['metric']:<16}{r['n']:>4}{r['before']:>8.3f}{r['after']:>8.3f}{r['diff']:>+8.3f}{interval:>20}{r['improved']:>8}{r['worsened']:>7}  {r['verdict']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-only", action="store_true", help="only list the chunks the BM25 index would be built from")
    args = parser.parse_args()
    setup_logging(level="WARNING")

    env = get_settings()
    settings = env.model_copy(update={"rerank_enabled": False, "retrieval_max_chunks_per_doc": None})
    if env.rerank_enabled or env.retrieval_max_chunks_per_doc:
        print("Note: RERANK_ENABLED / RETRIEVAL_MAX_CHUNKS_PER_DOC from the environment are overridden (off) for this experiment.")

    store = PineconeStore(settings)
    corpus = hybrid_ab.LexicalCorpus(store.iter_chunks())
    info = corpus.describe()
    print(f"Lexical corpus: {info['chunks']} chunks in {len(info['documents'])} documents")
    for name, n in info["documents"].items():
        print(f"  {n:>4}  {name}")
    kb_dir_files = sorted(p.name for p in settings.kb_documents_dir.iterdir() if p.is_file() and p.suffix.lower() in settings.allowed_extensions)
    if sorted(info["documents"]) != kb_dir_files:
        print(f"WARNING: the index holds {sorted(set(info['documents']) ^ set(kb_dir_files))} that differ from data/kb_documents; results describe the index, not the repo corpus.")
    if args.corpus_only or not info["chunks"]:
        return 0 if info["chunks"] else 1

    cases = load_dataset(settings.eval_dataset_path)
    labelled = [c for c in cases if c.expected_documents]
    print(f"\n{len(labelled)} questions with expected documents; pool={settings.eval_retrieval_pool}; params={hybrid_ab.PARAMS}")
    retriever = Retriever(settings, embedder=get_embedding_service(), store=store)
    t0 = time.time()
    per_system = hybrid_ab.run_experiment(cases, retriever, corpus, settings)
    print(f"ranked {len(per_system['dense'])} questions x 3 systems in {time.time() - t0:.0f}s")

    split = load_split(settings.eval_split_path)
    subsets = {"dev": set(split.dev), "test": set(split.test)}
    report = hybrid_ab.summarize(per_system, cases, subsets, resamples=settings.eval_bootstrap_resamples, seed=settings.eval_bootstrap_seed, confidence=settings.eval_confidence)
    report |= {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "corpus": info, "dataset_sha256": file_sha256(settings.eval_dataset_path),
        "settings": {"embedding_model": settings.embedding_model_name, "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap, "eval_ks": list(settings.eval_ks), "pool": settings.eval_retrieval_pool},
    }

    # Sanity check: the dense arm should reproduce the committed retrieval baseline when the setup is the same.
    try:
        baseline = regression.load_baseline(settings.eval_retrieval_baseline_path)
        ctx = regression.context(settings, list(info["documents"]))
        if ctx == baseline["context"]:
            dense = regression.aggregate(per_system["dense"], sorted(per_system["dense"]), regression.compared_keys(settings.eval_ks))
            diff = max(abs((dense[k] or 0) - (baseline["metrics"].get(k) or 0)) for k in dense)
            report["dense_vs_baseline"] = {"comparable": True, "max_abs_difference": diff}
            print(f"\nDense arm vs committed retrieval baseline: max |difference| = {diff:.4f} ({'reproduces it' if diff < 0.005 else 'DOES NOT reproduce it; check the index'})")
        else:
            changed = sorted(k for k in set(ctx) | set(baseline["context"]) if ctx.get(k) != baseline["context"].get(k))
            report["dense_vs_baseline"] = {"comparable": False, "differs": changed}
            print(f"\nDense arm vs committed retrieval baseline: not comparable (setup differs in {changed}).")
    except ValueError as exc:
        print(f"\nNo retrieval baseline to compare the dense arm with ({exc}).")

    for subset in ("all", "dev", "test"):
        _print_means(report, subset)
    for subset in ("all", "dev", "test"):
        _print_paired(report, subset, "hybrid")
    _print_paired(report, "all", "bm25")
    ch = report["changed_vs_dense_by_doc_mrr"]
    print(f"\nHybrid vs dense by document MRR: improved on {len(ch['improved'])} question(s) {ch['improved']}, worsened on {len(ch['worsened'])} {ch['worsened']}")
    d = report["decision"]
    print(f"\nPRE-REGISTERED DECISION (all questions, document metrics): {d['outcome']}\n  {d['reason']}\n  rule: {d['rule']}")
    for caveat in report["caveats"]:
        print(f"  - {caveat}")

    out_dir = settings.eval_dir / "results" / time.strftime("hybrid_ab_%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics = [*hybrid_ab.DOC_METRICS, *hybrid_ab.CHUNK_METRICS]
    by_id = {c.id: c for c in cases}
    with open(out_dir / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "category", "split", "question", *[f"{s}_{m}" for s in hybrid_ab.SYSTEMS for m in metrics]])
        w.writeheader()
        for cid in sorted(per_system["dense"]):
            w.writerow({"case_id": cid, "category": by_id[cid].category, "split": "test" if cid in split.test else "dev", "question": by_id[cid].question,
                        **{f"{s}_{m}": per_system[s][cid].get(m) for s in hybrid_ab.SYSTEMS for m in metrics}})
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
