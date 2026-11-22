"""Check the setup before a token-spending evaluation run. No Groq or Tavily calls, no tokens.

It reads the settings and keys (presence only, never the values), loads and validates the dataset, compares the SQLite documents
table and the Pinecone index with the core knowledge-base files, and (with --resume or --like) compares today's settings with a
saved run, because a resumed run keeps the settings it started with in config.json while the answers come from today's. It
cannot see whether Docker is stopped or whether anyone else is using the Groq key; it prints those as reminders.

Usage (project root, .venv active, the same environment variables you will run with):
    python scripts/preflight_eval.py                           # the main dataset, whole run
    python scripts/preflight_eval.py --split test              # before `run_eval.py --split test`
    python scripts/preflight_eval.py --resume fresh-1          # before `run_eval.py --resume fresh-1`
    python scripts/preflight_eval.py --like fresh-1 --split test    # will the new run pool with fresh-1?
    python scripts/preflight_eval.py --skip-index              # SQLite only (no Pinecone call)
For the fresh block set EVAL_DATASET_PATH and EVAL_SPLIT_PATH first (docs/RUN_PLAN.md, Day 1).
Exit code 0 = READY (warnings allowed), 1 = NOT READY.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import Settings, get_settings  # noqa: E402
from copilot.evaluation import preflight as pf  # noqa: E402
from copilot.evaluation.schemas import load_dataset, validate_dataset  # noqa: E402
from copilot.evaluation.split import load_split  # noqa: E402
from copilot.evaluation.system import settings_snapshot  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402

RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
REMINDERS = (
    "Docker is stopped (`docker compose down`) and no API or UI is answering questions.",
    "Nobody else is using the Groq key; the PowerShell window stays open and the PC stays awake.",
    "Groq reachability and the remaining daily quota: `python scripts\\check_groq.py` (spends a few tokens).",
)


def _has(secret) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


def read_index(settings: Settings) -> tuple[dict[str, int] | None, str | None]:
    """Chunks per document name in the Pinecone namespace, or ``(None, error kind)``; the message is never printed."""
    try:
        from copilot.vectorstore.pinecone_store import PineconeStore

        return dict(Counter(c.document_name for c in PineconeStore(settings).iter_chunks())), None
    except Exception as exc:  # noqa: BLE001 - report the kind of failure only; a message could echo configuration
        return None, type(exc).__name__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all", help="the --split you are about to run (default all)")
    ap.add_argument("--resume", metavar="RUN_ID", help="check before resuming this run")
    ap.add_argument("--like", metavar="RUN_ID", help="check that today's settings match this earlier run (so the two can be pooled)")
    ap.add_argument("--skip-index", action="store_true", help="do not read the Pinecone index")
    args = ap.parse_args(argv)

    setup_logging(level="WARNING")
    settings = get_settings()
    snapshot = settings_snapshot(settings)
    checks = pf.check_settings({**snapshot, "has_groq_key": _has(settings.groq_api_key), "has_pinecone_key": _has(settings.pinecone_api_key), "has_tavily_key": _has(settings.tavily_api_key)})

    path = settings.eval_dataset_path
    print(f"Dataset: {path}\nPlanned split: {args.split}\n")
    cases = []
    if not path.is_file():
        checks.append(pf.Check("dataset valid", "fail", f"dataset not found: {path}"))
    else:
        try:
            cases = load_dataset(path)
            checks.append(pf.check_dataset(str(path), len(cases), validate_dataset(cases, settings.kb_documents_dir)))
            checks.append(pf.check_fingerprint(pf.cases_fingerprint(path), pf.expected_fingerprint(path)))
        except (ValueError, KeyError, OSError) as exc:
            checks.append(pf.Check("dataset valid", "fail", f"cannot read the dataset ({type(exc).__name__})"))

    split_problems: list[str] = []
    split_ids: list[str] = []
    split_exists = settings.eval_split_path.is_file()
    if split_exists:
        try:
            sp = load_split(settings.eval_split_path)
            split_problems, split_ids = sp.problems({c.id: c.category for c in cases}), list(sp.assignment())
        except ValueError as exc:
            split_problems = [str(exc)]
    checks.append(pf.check_split(args.split, split_exists, split_problems, [c.id for c in cases], split_ids))

    core = sorted(f.name for f in settings.kb_documents_dir.iterdir() if f.is_file() and f.suffix.lower() in settings.allowed_extensions) if settings.kb_documents_dir.is_dir() else []
    rows = pf.read_documents(settings.sqlite_path)
    if rows is None:
        checks.append(pf.Check("SQLite documents", "fail", f"cannot read the documents table in {settings.sqlite_path} (run scripts\\init_db.py and scripts\\ingest_kb.py)"))
        rows = []
    else:
        checks += pf.check_documents(rows, core)
    counts, error = (None, None) if args.skip_index else read_index(settings)
    checks += pf.check_index(counts, error, rows, core)

    results_dir = settings.eval_dir / "results"
    for flag, run_id in (("resume", args.resume), ("like", args.like)):
        if not run_id:
            continue
        if not RUN_ID.fullmatch(run_id):
            checks.append(pf.Check(f"--{flag}", "fail", "invalid run id (letters, digits, '_' and '-', at most 64 characters)"))
            continue
        exists, saved, status = pf.saved_run(results_dir, run_id)
        if flag == "resume":
            checks.append(pf.check_resumable(run_id, exists, status))
            consequence = "a resumed run keeps the settings it started with in config.json while the new answers would use today's; restore them or start a new run"
        else:
            exists = exists or bool(saved)
            consequence = "scripts/render_results.py --pool would refuse to combine the two runs"
        if exists and saved:
            checks.append(pf.compare_settings(saved, snapshot, label=f"settings vs {run_id}", consequence=consequence))
        elif flag == "like":
            checks.append(pf.Check(f"settings vs {run_id}", "fail", "no saved run with settings found"))

    print(pf.render(checks))
    print("\nNot checked here:")
    for r in REMINDERS:
        print(f"  - {r}")
    ready, sentence = pf.verdict(checks)
    print(f"\n{sentence}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
