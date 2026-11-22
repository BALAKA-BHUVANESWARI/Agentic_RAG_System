"""Print the README results tables from saved runs. Reads summary.json only: no Groq, Tavily, Pinecone or network calls, no tokens.

Usage (project root):
    python scripts/render_results.py 20261119T120610Z-f01dbc
    python scripts/render_results.py fresh-1 test-1 --pool            # each run, then the two pooled
    python scripts/render_results.py fresh-1 --allow-partial          # a run that has not finished yet (banner added)
    python scripts/render_results.py fresh-1 test-1 --pool --out docs/RESULTS_TABLES.md

RUN is a run id under data/eval/results, a run folder, or a summary.json path. Nothing is written unless --out is given.
Exit code 2 when a run is unfinished (without --allow-partial) or runs cannot be pooled; the reason is printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.evaluation.results_table import load_run_view, render_many  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run id, run folder or summary.json path")
    ap.add_argument("--pool", action="store_true", help="also print the runs pooled (counts added; refuses runs whose settings differ or that repeat the same questions)")
    ap.add_argument("--allow-partial", action="store_true", help="render a run that did not finish, with a PARTIAL banner")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "data" / "eval" / "results")
    ap.add_argument("--out", type=Path, help="write the Markdown here as well as printing it")
    args = ap.parse_args(argv)
    try:
        runs = [load_run_view(r, args.results_dir) for r in args.runs]
        text = render_many(runs, pooled=args.pool, allow_partial=args.allow_partial)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
