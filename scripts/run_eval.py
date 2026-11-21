"""Run the evaluation: baseline RAG and/or the agentic workflow over the labelled dataset.

Examples (from the project root, venv active):
    python scripts/run_eval.py --validate-only                      # check dataset vs knowledge base (no network, no cost)
    python scripts/run_eval.py --per-category 1 --no-judge          # cheap smoke run: 8 questions x 2 systems
    python scripts/run_eval.py --mode both                          # full run (93 questions x 2 systems; slow on free Groq tiers)
    python scripts/run_eval.py --categories direct_kb paraphrased --mode agentic
    python scripts/run_eval.py --resume <run_id>                    # continue an interrupted run / retry failed cases
    python scripts/run_eval.py --list-runs
    python scripts/run_eval.py --split dev --per-category 3         # tune on the development split
    python scripts/run_eval.py --split test                         # the frozen test split: report it, do not tune on it
    python scripts/run_eval.py --attacks-only --no-judge            # only the 20 adversarial questions, both systems
    python scripts/run_eval.py --list-variants                      # the ablation ladder (no network, no cost)
    python scripts/run_eval.py --variants baseline,grade,agentic --per-category 1 --no-judge   # ablation smoke run
    python scripts/run_eval.py --variants all --run-id ablation-1   # every rung; ~6x the tokens of one system, resumable

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
from copilot.evaluation.split import load_split  # noqa: E402
from copilot.evaluation.store import EvalRunStore  # noqa: E402
from copilot.evaluation.variants import LADDER, parse_variants  # noqa: E402
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


def _pts(x: float) -> str:
    return f"{x * 100:+.1f}"


def _print_intervals(iv: dict | None) -> None:
    """Rates with their intervals, then every paired difference with the cases that flipped."""
    if not iv:
        return
    pct = f"{iv['confidence']:.0%}"
    print(f"\n{pct} intervals (Wilson per rate; paired bootstrap over questions for differences; {iv['caveat']})")
    for mode, rates in iv["rates"].items():
        r = rates.get("overall_correct")
        if r:
            print(f"  {mode:<10} overall correct {r['value']:.0%} [{r['lo']:.0%}, {r['hi']:.0%}]  (n={r['den']})")
    for p in iv["paired"]:
        print(f"\n  {p['from']} -> {p['to']}  (same {p['n_cases']} questions)")
        for m in p["metrics"]:
            print(f"    {m['label']:<30} {_pts(m['diff'])} pts [{_pts(m['lo'])}, {_pts(m['hi'])}]  gained {m['gained']} / lost {m['lost']}  n={m['n']}  -> {m['verdict']}")


def _print_attacks(att: dict | None) -> None:
    """Attack-success rate on the adversarial block, per system, with the successful attacks listed."""
    if not att:
        return
    print(f"\nAttack success on the {att['n_attacks']} adversarial questions (lower is better). {att['caveat']}")
    for mode, a in att["modes"].items():
        r = a["asr"]
        ci = f" [{r['lo']:.0%}, {r['hi']:.0%}]" if r["lo"] is not None else ""
        print(f"  {mode:<10} success {_fmt(r)}{ci}  leaks {a['by_flag']['leak']}, answered-should-refuse {a['by_flag']['answered_when_should_abstain']}, "
              f"unwanted web search {a['by_flag']['unwanted_web_search']}; held but answer incorrect {a['held_but_incorrect']}; not judged (errors) {a['not_judged_errors']}")
        for c in a["succeeded_cases"]:
            print(f"      {c['case_id']:<5} {','.join(c['flags']):<40} {c['question'][:50]}")


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
    w = 22 if len(modes) <= 2 else 14  # an ablation run has up to six columns
    print("\n" + f"{'metric':<32}" + "".join(f"{m:>{w}}" for m in modes))
    for label, section, key in rows:
        print(f"{label:<32}" + "".join(f"{_fmt(summary['modes'][m].get(section, {}).get(key)):>{w}}" for m in modes))
    for label, key in (("avg latency (ms)", "latency_avg_ms"), ("p95 latency (ms)", "latency_p95_ms"), ("total tokens", "total_tokens"), ("est. cost (USD)", "estimated_cost_usd")):
        print(f"{label:<32}" + "".join(f"{_fmt(summary['modes'][m]['system'].get(key)):>{w}}" for m in modes))
    if summary.get("comparison"):
        print("\nBaseline vs agentic:")
        for line in summary["comparison"]["findings"]:
            print("  -", line)
    if summary.get("ablation"):
        print("\nAblation (each step adds one agent component; see docs/ABLATION.md):")
        for line in summary["ablation"]["findings"]:
            print("  -", line)
    _print_intervals(summary.get("intervals"))
    _print_attacks(summary.get("attacks"))
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
    p.add_argument("--split", choices=["all", "dev", "test"], default="all", help="frozen dev/test split (docs/EVAL_SPLIT.md); default runs every case")
    p.add_argument("--variants", metavar="NAMES", help="ablation rungs instead of --mode: comma list or 'all' (see --list-variants)")
    p.add_argument("--list-variants", action="store_true", help="show the ablation ladder and exit")
    p.add_argument("--limit", type=int, help="first N cases (after the other filters)")
    p.add_argument("--per-category", type=int, help="first N cases of EACH category")
    p.add_argument("--categories", nargs="+", choices=CATEGORIES, help="only these categories")
    p.add_argument("--attacks-only", action="store_true", help="only the 20 adversarial questions (X01-X20): an attack-success run")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (faster, fewer tokens)")
    p.add_argument("--delay", type=float, help="seconds to pause between requests (overrides EVAL_CASE_DELAY_S)")
    p.add_argument("--run-id", help="name for a new run")
    p.add_argument("--resume", metavar="RUN_ID", help="resume an earlier run; finished cases are skipped, failed ones retried")
    p.add_argument("--validate-only", action="store_true", help="validate the dataset against the knowledge base and exit")
    p.add_argument("--list-runs", action="store_true", help="list earlier runs and exit")
    args = p.parse_args()

    setup_logging(level="WARNING")
    settings = get_settings()

    if args.list_variants:
        for v in LADDER:
            print(f"{v.name:<9} {v.label:<32} {v.description}")
            if v.overrides:
                print(f"{'':<9} overrides: " + ", ".join(f"{k}={val}" for k, val in v.overrides.items()))
        return 0
    variants = None
    if args.variants:
        if args.mode != "both":
            print("Use either --mode or --variants, not both.")
            return 2
        try:
            variants = parse_variants(args.variants)
        except ValueError as exc:
            print("Cannot start:", exc)
            return 2

    if args.validate_only:
        cases = load_dataset(settings.eval_dataset_path)
        problems = validate_dataset(cases, settings.kb_documents_dir)
        print(f"{len(cases)} cases in {settings.eval_dataset_path}")
        if settings.eval_split_path.exists():
            try:
                sp = load_split(settings.eval_split_path)
                problems += [f"split: {q}" for q in sp.problems({c.id: c.category for c in cases})]
                print(f"Split: {len(sp.test)} test / {len(sp.dev)} dev ({settings.eval_split_path})")
            except ValueError as exc:
                problems.append(f"split: {exc}")
        else:
            print(f"No split file at {settings.eval_split_path} (run scripts/freeze_split.py --create).")
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
        mode=args.mode, variants=variants, split=args.split, attacks_only=args.attacks_only, limit=args.limit, per_category=args.per_category, categories=args.categories, use_judge=not args.no_judge,
        run_id=args.resume or args.run_id, resume=bool(args.resume), case_delay_s=args.delay,
    )
    try:
        resolved, cases = runner.prepare(opts)
    except ValueError as exc:
        print("Cannot start:", exc)
        return 2
    print(f"Evaluating {len(cases)} case(s) [{resolved.split} split] x {len(resolved.modes)} system(s); judge={'on' if resolved.use_judge else 'off'}. This calls Groq/Pinecone"
          f"{'/Tavily' if any(m in resolved.modes for m in ('agentic', 'external')) and settings.external_search_enabled else ''} for real.")
    summary = runner.run(opts, progress=lambda d, t, m: print(f"[{d:>3}/{t}] {m}", flush=True), should_cancel=lambda: stop["flag"])
    _print_summary(summary, runner.run_dir(summary["run_id"]))
    return 0 if summary["status"] in {"completed", "cancelled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
