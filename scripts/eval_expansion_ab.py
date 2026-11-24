"""Experiment C, dev-only step: choose between one and two paraphrases by document retrieval metrics. Spends a few Groq tokens.

Design and rules: docs/POST_DAY3_EXPERIMENTS.md (Amendment 3). Uses the existing Pinecone index (read-only queries) and the local embedder.
One Groq call per dev question that names expected documents (about 300 tokens each) writes two paraphrases; they are saved to
``paraphrases.json`` so a rerun with ``--paraphrases`` costs nothing and is reproducible. The reranker and the per-document cap are forced off.

Usage (project root, venv active, the 11 core documents ingested):
    python scripts/eval_expansion_ab.py --delay 2
    python scripts/eval_expansion_ab.py --paraphrases data/eval/results/<folder>/paraphrases.json     # no tokens
Output: printed tables + data/eval/results/expansion_ab_<timestamp>/{summary.json,paraphrases.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.embeddings.service import get_embedding_service  # noqa: E402
from copilot.evaluation import expansion_ab  # noqa: E402
from copilot.evaluation.schemas import load_dataset  # noqa: E402
from copilot.evaluation.split import load_split  # noqa: E402
from copilot.graph.expansion import expand_query  # noqa: E402
from copilot.llm.client import GroqClient  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402
from copilot.vectorstore.pinecone_store import PineconeStore  # noqa: E402
from copilot.vectorstore.retriever import Retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paraphrases", help="reuse a saved paraphrases.json (no Groq call)")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between Groq calls (free-tier rate limit)")
    args = parser.parse_args()
    setup_logging(level="WARNING")

    env = get_settings()
    settings = env.model_copy(update={"rerank_enabled": False, "retrieval_max_chunks_per_doc": None})
    cases = load_dataset(settings.eval_dataset_path)
    split = load_split(settings.eval_split_path)
    dev_ids = set(split.dev)
    dev = [c for c in cases if c.id in dev_ids and c.expected_documents]
    print(f"{len(dev)} dev questions with expected documents (the test split and the fresh blocks are NOT used)")

    if args.paraphrases:
        paraphrases = json.loads(Path(args.paraphrases).read_text(encoding="utf-8"))
    else:
        llm = GroqClient(settings)
        paraphrases, tokens = {}, 0
        for c in dev:
            got, usage = expand_query(llm, c.question, 2)
            paraphrases[c.id] = got
            tokens += usage.total_tokens if usage else 0
            time.sleep(args.delay)
        print(f"Generated paraphrases for {len(paraphrases)} questions; about {tokens} tokens")

    retriever = Retriever(settings, embedder=get_embedding_service(), store=PineconeStore(settings))
    per_arm = expansion_ab.run_arms(dev, retriever, list(settings.eval_ks), settings.eval_retrieval_pool, paraphrases)
    ids = sorted(per_arm["control"])
    choice = expansion_ab.choose_arm(per_arm, ids, resamples=settings.eval_bootstrap_resamples, seed=settings.eval_bootstrap_seed, confidence=settings.eval_confidence)

    print(f"\n{'metric':<16}" + "".join(f"{a:>10}" for a in expansion_ab.ARMS))
    for m in expansion_ab.DOC_METRICS:
        cells = []
        for arm in expansion_ab.ARMS:
            vals = [per_arm[arm][i].get(m) for i in ids if per_arm[arm][i].get(m) is not None]
            cells.append(sum(vals) / len(vals) if vals else float("nan"))
        print(f"{m:<16}" + "".join(f"{v:>10.3f}" for v in cells))
    for arm in ("para1", "para2"):
        print(f"\n{arm} vs control (paired bootstrap; verdict only if the interval excludes 0 and n >= 10)")
        for r in choice["arms"][arm]["rows"]:
            print(f"  {r['metric']:<14} n={r['n']:>3} diff {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] better {r['improved']} worse {r['worsened']}  {r['verdict']}")
    print(f"\nPRE-REGISTERED CHOICE: {choice['outcome']} -> {choice['chosen']}\n  {choice['reason']}")

    out_dir = settings.eval_dir / "results" / time.strftime("expansion_ab_%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paraphrases.json").write_text(json.dumps(paraphrases, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"choice": choice, "per_arm": per_arm, "questions": ids, "settings": {"embedding_model": settings.embedding_model_name, "pool": settings.eval_retrieval_pool}}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
