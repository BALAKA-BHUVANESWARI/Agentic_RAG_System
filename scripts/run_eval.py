"""Run the evaluation: baseline RAG and/or the agentic workflow over the labelled dataset.

Examples (from the project root, venv active):
    python scripts/run_eval.py --validate-only                      # check dataset vs knowledge base (no network, no cost)
    python scripts/run_eval.py --per-category 1 --no-judge          # cheap smoke run: 8 questions x 2 systems
    python scripts/run_eval.py --mode both                          # full run (52 questions x 2 systems; slow on free Groq tiers)
    python scripts/run_eval.py --categories direct_kb paraphrased --mode agentic
    python scripts/run_eval.py --resume <run_id>                    # continue an interrupted run / retry failed cases
    python scripts/run_eval.py --list-runs

Press Ctrl+C once to stop after the current case (the run stays resumable); twice to abort immediately.
Results: data/eval/results/<run_id>/ (summary.json, cases.csv, metrics.csv, cases.jsonl, retrieval.jsonl).
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.db.schema import init_db  # noqa: E402
from copilot.evaluation.runner import RunOptions  # noqa: E402
from copilot.evaluation.schemas import CATEGORIES, load_dataset, validate_dataset  # noqa: E402
from copilot.evaluation.store import EvalRunStore  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402


def _fmt(metric) -> str:
    """Format a ``{"value","num","den"}`` rate, a float or None."""
    if isinstance(metric, dict):
        if metric.get("value") is None:
            return "n/a"
        return f"{metric['value']:.0%} ({metric['num']}/{metric['den']})"
    if metric is None:
        return "n/a"
    return f"{metric:.3f}" if isinstance(metric, float) else str(metric)


def _print_summary(summary: dict, run_dir: Path) -> None:
    print("\n" + "=" * 78)
    print(f"Run {summary['run_id']}  status={summary['status']}  cases={summary['dataset']['num_cases']}  "
          f"units ok={summary['progress']['units_ok']}/{summary['progress']['units_total']}")
    if summary.get("error"):
        print("ERROR:", summary["error"])
    retr = summary.get("retrieval")
    if retr:
        m = retr["metrics"]
        k = max(retr["ks"])
        print(f"\nRetrieval (n={retr['n_cases']}):  doc recall@{k}={_fmt(m.get(f'doc_recall@{k}'))}  doc MRR={_fmt(m.get('doc_mrr'))}  "
              f"chunk precision@{k}={_fmt(m.get(f'chunk_precision@{k}'))}  chunk nDCG@{k}={_fmt(m.get(f'chunk_ndcg@{k}'))}")
    rows = [
        ("overall correct", "answer", "overall_correct_rate"), ("answerable: correct", "answer", "answer_correct_rate"),
        ("false abstentions", "answer", "false_abstention_rate"), ("correct abstentions", "answer", "correct_abstention_rate"),
        ("hallucinated (should abstain)", "answer", "hallucinated_answer_rate"), ("citation correct", "answer", "citation_correct_rate"),
        ("faithful (judge)", "answer", "faithfulness_rate"), ("private-first routing", "routing", "private_first_routing_accuracy"),
        ("unnecessary external search", "routing", "unnecessary_external_search_rate"), ("external fallback accuracy", "routing", "external_fallback_accuracy"),
        ("rewrite success", "routing", "query_rewrite_success_rate"), ("evidence grading accuracy", "routing", "evidence_grading_accuracy"),
        ("validation failures", "routing", "final_validation_failure_rate"), ("error rate", "system", "error_rate"),
    ]
    modes = list(summary["modes"])
    print("\n" + f"{'metric':<32}" + "".join(f"{m:>22}" for m in modes))
    for label, section, key in rows:
        print(f"{label:<32}" + "".join(f"{_fmt(summary['modes'][m].get(section, {}).get(key)):>22}" for m in modes))
    for label, key in (("avg latency (ms)", "latency_avg_ms"), ("p95 latency (ms)", "latency_p95_ms"), ("total tokens", "total_tokens"), ("est. cost (USD)", "estimated_cost_usd")):
        print(f"{label:<32}" + "".join(f"{_fmt(summary['modes'][m]['system'].get(key)):>22}" for m in modes))
    if summary.get("comparison"):
        print("\nBaseline vs agentic:")
        for line in summary["comparison"]["findings"]:
            print("  -", line)
    ing = summary.get("ingestion")
    if ing:
        print(f"\nIngestion success: attempts {_fmt(ing['attempts'])}, documents currently indexed {_fmt(ing['documents'])}")
    failed = summary.get("failed_cases", [])
    print(f"\nFailed or incorrect cases: {len(failed)}")
    for f in failed[:15]:
        print(f"  {f['case_id']:<5} {f['mode']:<9} {f['outcome']:<18} {f['question'][:60]}")
    if len(failed) > 15:
        print(f"  ... {len(failed) - 15} more in summary.json")
    for n in summary.get("notes", []):
        print("NOTE:", n)
    print(f"\nFiles: {run_dir}")


def main() -> int:
    """Parse arguments and run."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["both", "agentic", "baseline"], default="both")
    p.add_argument("--limit", type=int, help="first N cases (after the other filters)")
    p.add_argument("--per-category", type=int, help="first N cases of EACH category")
    p.add_argument("--categories", nargs="+", choices=CATEGORIES, help="only these categories")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (faster, fewer tokens)")
    p.add_argument("--delay", type=float, help="seconds to pause between requests (overrides EVAL_CASE_DELAY_S)")
    p.add_argument("--run-id", help="name for a new run")
    p.add_argument("--resume", metavar="RUN_ID", help="resume an earlier run; finished cases are skipped, failed ones retried")
    p.add_argument("--validate-only", action="store_true", help="validate the dataset against the knowledge base and exit")
    p.add_argument("--list-runs", action="store_true", help="list earlier runs and exit")
    args = p.parse_args()

    setup_logging(level="WARNING")
    settings = get_settings()

    if args.validate_only:
        cases = load_dataset(settings.eval_dataset_path)
        problems = validate_dataset(cases, settings.kb_documents_dir)
        print(f"{len(cases)} cases in {settings.eval_dataset_path}")
        for q in problems:
            print("PROBLEM:", q)
        print("Dataset is valid." if not problems else f"{len(problems)} problem(s) found.")
        return 1 if problems else 0

    init_db(settings.sqlite_path)
    if args.list_runs:
        for r in EvalRunStore(settings.sqlite_path).list(50):
            print(f"{r['run_id']:<28} {r['status']:<10} mode={r['mode']:<9} cases={r['num_cases']:<3} started={r['started_at']}")
        return 0

    from copilot.evaluation.factory import build_graph_deps, build_runner
    from copilot.graph.runner import AgenticRAG

    for name in ("groq", "pinecone"):
        settings.require_secret(name)  # clear, key-free error if missing
    deps = build_graph_deps(settings)
    runner = build_runner(settings, deps, AgenticRAG(settings, deps))

    stop = {"flag": False}

    def on_sigint(signum, frame):  # noqa: ARG001
        stop["flag"] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # a second Ctrl+C aborts immediately
        print("\nStopping after the current case (press Ctrl+C again to abort immediately)...")

    signal.signal(signal.SIGINT, on_sigint)

    opts = RunOptions(
        mode=args.mode, limit=args.limit, per_category=args.per_category, categories=args.categories, use_judge=not args.no_judge,
        run_id=args.resume or args.run_id, resume=bool(args.resume), case_delay_s=args.delay,
    )
    try:
        resolved, cases = runner.prepare(opts)
    except ValueError as exc:
        print("Cannot start:", exc)
        return 2
    print(f"Evaluating {len(cases)} case(s) x {len(resolved.modes)} system(s); judge={'on' if resolved.use_judge else 'off'}. This calls Groq/Pinecone"
          f"{'/Tavily' if 'agentic' in resolved.modes and settings.external_search_enabled else ''} for real.")
    summary = runner.run(opts, progress=lambda d, t, m: print(f"[{d:>3}/{t}] {m}", flush=True), should_cancel=lambda: stop["flag"])
    _print_summary(summary, runner.run_dir(summary["run_id"]))
    return 0 if summary["status"] in {"completed", "cancelled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
