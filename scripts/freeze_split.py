"""Create, extend and check the frozen dev/test split of the evaluation dataset (no network, no cost).

    python scripts/freeze_split.py --check                 # default: integrity of data/eval/split.json against the dataset
    python scripts/freeze_split.py --show                  # cases per category in dev and test
    python scripts/freeze_split.py --create                # first freeze; refuses to overwrite an existing split
    python scripts/freeze_split.py --add-new               # assign cases added to the dataset later; old assignments never move
    python scripts/freeze_split.py --refresh-fingerprint   # ONLY after deliberately moving a burned case to dev (see docs/EVAL_SPLIT.md)

The rule and the policy are written into split.json and explained in docs/EVAL_SPLIT.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.evaluation import split as sp  # noqa: E402

DATASET = ROOT / "data" / "eval" / "dataset.json"
SPLIT = ROOT / "data" / "eval" / "split.json"
DEFAULT_SALT = "it-support-copilot-v1"


def _dataset(path: Path) -> dict[str, str]:
    """``{id: category}`` read straight from the JSON (no model validation needed here)."""
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit("The dataset has duplicate case ids.")
    return {c["id"]: c["category"] for c in cases}


def _write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(sp.dumps(doc), encoding="utf-8")
    tmp.replace(path)


def _show(dataset: dict[str, str], split: sp.Split) -> None:
    table: dict[str, Counter] = {}
    for cid, category in dataset.items():
        table.setdefault(sp.stratum_of(cid, category), Counter())["test" if cid in split.test else "dev" if cid in split.dev else "unassigned"] += 1
    print(f"{'stratum':<32}{'dev':>5}{'test':>6}{'unassigned':>12}")
    for name, c in sorted(table.items()):
        print(f"{name:<32}{c['dev']:>5}{c['test']:>6}{c['unassigned']:>12}")
    print(f"{'TOTAL':<32}{len(split.dev & set(dataset)):>5}{len(split.test & set(dataset)):>6}{len(set(dataset) - split.dev - split.test):>12}")


def main() -> int:
    """Parse arguments and run."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--show", action="store_true")
    g.add_argument("--create", action="store_true")
    g.add_argument("--add-new", action="store_true")
    g.add_argument("--refresh-fingerprint", action="store_true")
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--split-file", type=Path, default=SPLIT)
    p.add_argument("--salt", default=DEFAULT_SALT, help="only for --create; stored in the file")
    p.add_argument("--fraction", type=float, default=sp.DEFAULT_FRACTION, help="only for --create")
    args = p.parse_args()
    dataset = _dataset(args.dataset)

    if args.create:
        if args.split_file.exists():
            print(f"{args.split_file} already exists. The split is frozen; use --add-new for new cases.")
            return 2
        assignment = sp.plan(dataset.items(), salt=args.salt, fraction=args.fraction)
        _write(args.split_file, sp.build_document(assignment, salt=args.salt, fraction=args.fraction, created=date.today().isoformat()))
        print(f"Created {args.split_file}: {sum(1 for s in assignment.values() if s == 'test')} test / {sum(1 for s in assignment.values() if s == 'dev')} dev.")
        _show(dataset, sp.load_split(args.split_file))
        return 0

    try:
        split = sp.load_split(args.split_file)
    except ValueError as exc:
        print(exc)
        return 2

    if args.add_new:
        meta = split.meta
        merged = sp.plan(dataset.items(), split.assignment(), salt=meta["salt"], fraction=meta["fraction"])
        added = {i: s for i, s in merged.items() if i not in split.dev and i not in split.test}
        if not added:
            print("Nothing to add: every dataset case already has an assignment.")
            return 0
        _write(args.split_file, sp.build_document({**split.assignment(), **added}, salt=meta["salt"], fraction=meta["fraction"], created=meta["created"]))
        for cid, name in sorted(added.items()):
            print(f"  {cid:<6} -> {name}")
        print(f"Assigned {len(added)} new case(s); existing assignments were not touched.")
        return 0

    if args.refresh_fingerprint:
        doc = {**split.meta, "test": sorted(split.test), "dev": sorted(split.dev), "test_sha256": sp.fingerprint(split.test)}
        _write(args.split_file, doc)
        print("Fingerprint updated. Commit the change together with a note in docs/FINDINGS.md saying which case moved and why.")
        return 0

    if args.show:
        _show(dataset, split)
        return 0

    problems, warnings = split.problems(dataset), split.warnings(dataset)
    for w in warnings:
        print("WARNING:", w)
    for q in problems:
        print("PROBLEM:", q)
    print(f"Split: {len(split.test)} test / {len(split.dev)} dev, created {split.meta.get('created')}, salt {split.meta.get('salt')!r}.")
    print("Split is consistent with the dataset." if not problems else f"{len(problems)} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
