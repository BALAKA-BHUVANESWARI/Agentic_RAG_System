"""A frozen development / test split of the evaluation dataset.

Why: prompts, thresholds and routing rules were tuned while looking at evaluation failures. Numbers measured on the
very questions that drove those changes are optimistic. The split gives two honest uses of the same dataset:

* **dev**: look at failures, change things, re-run as often as you like.
* **test**: run it to REPORT a number. Never change the system because of an individual test failure. If a test failure
  does drive a change, move that case to dev in the same commit and say why in docs/FINDINGS.md (the case is "burned").

Rules that keep the split meaningful:

* The assignment is stratified by category (adversarial cases form their own stratum, id prefix ``X``) so every
  large-enough category has both dev and test questions, and it is decided by a salted hash, not by anyone's judgement
  of which questions are easy.
* It is stored in ``data/eval/split.json`` and **frozen**: existing assignments are never recomputed. New dataset
  cases get an assignment only through ``scripts/freeze_split.py --add-new``, which leaves the old ones untouched.
* A fingerprint of the test ids is stored, so an accidental edit of the file is caught by a unit test.

Honest limit: all 93 existing cases were visible while the system was developed, so today's test split is *less*
contaminated, not unseen. A truly clean test set needs fresh questions written after the freeze (see docs/EVAL_SPLIT.md).

This module uses only the standard library so it can run anywhere.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

SplitName = Literal["all", "dev", "test"]
FORMAT_VERSION = 1
DEFAULT_FRACTION = 0.3
MIN_STRATUM = 3  # strata smaller than this stay entirely in dev
ADVERSARIAL_PREFIX = "X"

RULE = (
    "Stratified by category (adversarial cases, ids starting with X, are their own stratum). Within a stratum the cases "
    "are ordered by sha256(salt:id) and the first round(fraction * size) become test (strata under 3 cases stay in dev). "
    "Existing assignments are frozen; new cases are assigned by scripts/freeze_split.py --add-new."
)
POLICY = (
    "Tune on dev. Run test to report a number. Do not change prompts, thresholds or routing because of an individual "
    "test failure; if you do, move that case to dev in the same commit and record why in docs/FINDINGS.md."
)


def stratum_of(case_id: str, category: str) -> str:
    """Stratum name: the category, with adversarial cases kept apart."""
    return f"{category}/adversarial" if case_id.startswith(ADVERSARIAL_PREFIX) else category


def _rank(salt: str, case_id: str) -> str:
    return hashlib.sha256(f"{salt}:{case_id}".encode("utf-8")).hexdigest()


def quota(size: int, fraction: float = DEFAULT_FRACTION) -> int:
    """How many of ``size`` cases of one stratum go to test."""
    return 0 if size < MIN_STRATUM else max(1, int(fraction * size + 0.5))


def fingerprint(test_ids: Iterable[str]) -> str:
    """SHA-256 of the sorted test ids (detects accidental edits of the stored file)."""
    return hashlib.sha256("\n".join(sorted(test_ids)).encode("utf-8")).hexdigest()


def plan(
    cases: Iterable[tuple[str, str]], frozen: Mapping[str, str] | None = None, *, salt: str, fraction: float = DEFAULT_FRACTION
) -> dict[str, str]:
    """Assignment ``{case_id: "dev" | "test"}`` for ``cases`` given as ``(id, category)``.

    Entries in ``frozen`` are kept as they are. Only unassigned cases are placed, topping each stratum up to its quota
    in hash order, so adding cases later never moves an old one.
    """
    frozen = frozen or {}
    strata: dict[str, list[str]] = {}
    for cid, category in cases:
        strata.setdefault(stratum_of(cid, category), []).append(cid)
    out: dict[str, str] = {}
    for ids in strata.values():
        have = sum(1 for i in ids if frozen.get(i) == "test")
        need = max(0, quota(len(ids), fraction) - have)
        free = sorted((i for i in ids if i not in frozen), key=lambda i: (_rank(salt, i), i))
        out.update({i: frozen[i] for i in ids if i in frozen})
        out.update({i: ("test" if pos < need else "dev") for pos, i in enumerate(free)})
    return out


def build_document(assignment: Mapping[str, str], *, salt: str, fraction: float, created: str) -> dict[str, Any]:
    """The JSON document stored in ``data/eval/split.json``."""
    test = sorted(i for i, s in assignment.items() if s == "test")
    dev = sorted(i for i, s in assignment.items() if s == "dev")
    return {
        "version": FORMAT_VERSION, "created": created, "salt": salt, "fraction": fraction, "rule": RULE, "policy": POLICY,
        "test_sha256": fingerprint(test), "test": test, "dev": dev,
    }


def dumps(doc: Mapping[str, Any]) -> str:
    """Stable text form (matches what ``freeze_split.py`` writes)."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class Split:
    """A loaded split."""

    test: frozenset[str]
    dev: frozenset[str]
    meta: Mapping[str, Any]

    def ids(self, name: SplitName) -> frozenset[str] | None:
        """Ids of a split (None for ``all``: no filtering)."""
        if name == "all":
            return None
        if name not in ("dev", "test"):
            raise ValueError(f"unknown split {name!r}; valid: all, dev, test")
        return self.test if name == "test" else self.dev

    def assignment(self) -> dict[str, str]:
        """``{id: split}`` for every assigned case."""
        return {**{i: "dev" for i in self.dev}, **{i: "test" for i in self.test}}

    def problems(self, dataset: Mapping[str, str]) -> list[str]:
        """Integrity problems against ``dataset`` (``{id: category}``). Any problem makes a split-based run refuse to start."""
        out: list[str] = []
        both = sorted(self.test & self.dev)
        if both:
            out.append(f"ids in both dev and test: {both[:5]}")
        unknown = sorted((self.test | self.dev) - set(dataset))
        if unknown:
            out.append(f"split lists ids that are not in the dataset (deleted or renamed?): {unknown[:5]}")
        unassigned = sorted(set(dataset) - self.test - self.dev)
        if unassigned:
            out.append(f"dataset cases without an assignment (run scripts/freeze_split.py --add-new): {unassigned[:5]}")
        if self.meta.get("test_sha256") != fingerprint(self.test):
            out.append("the test ids do not match the stored fingerprint: the file was edited by hand. If a case was moved on purpose "
                       "(burned), update test_sha256 with scripts/freeze_split.py --refresh-fingerprint and say why in docs/FINDINGS.md.")
        return out

    def warnings(self, dataset: Mapping[str, str]) -> list[str]:
        """Things worth a look that do not stop a run."""
        out: list[str] = []
        strata: dict[str, list[str]] = {}
        for cid, category in dataset.items():
            strata.setdefault(stratum_of(cid, category), []).append(cid)
        for name, ids in sorted(strata.items()):
            n_test = sum(1 for i in ids if i in self.test)
            if len(ids) >= MIN_STRATUM and (n_test == 0 or n_test == len(ids)):
                out.append(f"stratum {name!r} ({len(ids)} cases) has {n_test} test case(s)")
        share = len(self.test & set(dataset)) / len(dataset) if dataset else 0.0
        if dataset and not 0.2 <= share <= 0.4:
            out.append(f"test share is {share:.0%}, outside 20-40%")
        return out


def filter_cases(cases: list[Any], name: SplitName, path: Path | str) -> list[Any]:
    """Keep the cases of one split (``all`` keeps everything). Works on any objects that have an ``id``."""
    ids = load_split(path).ids(name)
    return list(cases) if ids is None else [c for c in cases if c.id in ids]


def load_split(path: Path | str) -> Split:
    """Read ``split.json`` (raises ``ValueError`` with a clear message if it is missing or malformed)."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Split file not found: {p} (create it with scripts/freeze_split.py --create)")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        test, dev = doc["test"], doc["dev"]
        if not (isinstance(test, list) and isinstance(dev, list) and all(isinstance(i, str) for i in test + dev)):
            raise TypeError("test and dev must be lists of ids")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Split file {p} is malformed: {exc}") from exc
    meta = {k: v for k, v in doc.items() if k not in ("test", "dev")}
    if len(set(test)) != len(test) or len(set(dev)) != len(dev):
        raise ValueError(f"Split file {p} lists an id twice")
    return Split(frozenset(test), frozenset(dev), meta)
