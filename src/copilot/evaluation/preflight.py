"""Pre-flight checks before a token-spending evaluation run (docs/RUN_PLAN.md, "Rules for every run").

A free-tier day is about 200,000 Groq tokens. A run on a polluted setup (reranker on, an extra document in the index, the
wrong dataset, settings changed between a run and its resume) wastes them and produces numbers that cannot be compared or
pooled. Every function here takes plain values and returns :class:`Check` records, so the logic is unit-testable without
Pinecone, SQLite or API keys; ``scripts/preflight_eval.py`` collects the real values. Nothing here calls Groq or Tavily.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from copilot.evaluation.results_table import STATISTICS_ONLY_SETTINGS

Status = Literal["ok", "warn", "fail", "skip"]
TAGS: dict[str, str] = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}
# One row of the SQLite ``documents`` table: (document_name, status, lifecycle, chunk_count).
DocumentRow = tuple[str, str, str, int]


@dataclass(frozen=True)
class Check:
    """One pre-flight result: ``fail`` blocks the run, ``warn`` does not, ``skip`` means it was not looked at."""

    name: str
    status: Status
    detail: str


# ------------------------------------------------------------------------------ dataset fingerprint
def cases_fingerprint(path: Path | str) -> str:
    """SHA-256 of a dataset's ``cases`` in canonical JSON (the recipe behind ``data/eval/fresh_dataset.sha256``)."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    return hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def expected_fingerprint(dataset_path: Path | str) -> str | None:
    """The sidecar ``<dataset>.sha256`` next to a dataset file, or None (only the fresh block has one)."""
    sidecar = Path(dataset_path).with_suffix(".sha256")
    return sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else None


# ------------------------------------------------------------------------------ reading what exists
def read_documents(db_path: Path | str) -> list[DocumentRow] | None:
    """Rows of the SQLite ``documents`` table through a read-only connection (None when it cannot be read)."""
    path = Path(db_path)
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT document_name, status, lifecycle, chunk_count FROM documents ORDER BY document_name").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return [(str(a), str(b), str(c), int(d)) for a, b, c, d in rows]


def saved_run(results_dir: Path | str, run_id: str) -> tuple[bool, dict[str, Any], str | None]:
    """``(has config.json, saved settings, status)`` for a run folder; settings come from config.json, else summary.json."""
    folder = Path(results_dir) / run_id
    config, summary = folder / "config.json", folder / "summary.json"
    saved: dict[str, Any] = {}
    status: str | None = None
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        status, saved = data.get("status"), dict(data.get("settings") or {})
    if config.is_file():
        saved = dict(json.loads(config.read_text(encoding="utf-8")).get("settings") or saved)
    return config.is_file(), saved, status


# ------------------------------------------------------------------------------ the checks
def check_settings(v: Mapping[str, Any]) -> list[Check]:
    """Settings that must not differ from the recorded runs. ``v`` is ``settings_snapshot`` plus ``has_*_key`` booleans."""
    out: list[Check] = []

    def add(name: str, ok: bool, good: str, bad: str, bad_status: Status = "fail") -> None:
        out.append(Check(name, "ok" if ok else bad_status, good if ok else bad))

    cap = v.get("retrieval_max_chunks_per_doc")
    add("reranker off", not v.get("rerank_enabled"), "RERANK_ENABLED is false",
        "RERANK_ENABLED is true: results would not be comparable with the recorded runs (docs/FINDINGS.md, finding 4)")
    add("no per-document chunk cap", cap is None, "RETRIEVAL_MAX_CHUNKS_PER_DOC is not set",
        f"RETRIEVAL_MAX_CHUNKS_PER_DOC is {cap}; it stays off until a run shows it does no harm")
    add("full agent", bool(v.get("agent_use_grader")) and bool(v.get("agent_use_validator")), "grader and validator are on",
        "AGENT_USE_GRADER or AGENT_USE_VALIDATOR is false in the environment, so the 'agentic' system would be an ablation rung")
    add("web search enabled", bool(v.get("external_search_enabled")), "EXTERNAL_SEARCH_ENABLED is true",
        "EXTERNAL_SEARCH_ENABLED is false: the current-information questions can only abstain", "warn")
    add("Groq key", bool(v.get("has_groq_key")), "GROQ_API_KEY is set", "GROQ_API_KEY is missing")
    add("Pinecone key", bool(v.get("has_pinecone_key")), "PINECONE_API_KEY is set", "PINECONE_API_KEY is missing")
    add("Tavily key", bool(v.get("has_tavily_key")) or not v.get("external_search_enabled"), "TAVILY_API_KEY is set (or web search is off)",
        "TAVILY_API_KEY is missing while web search is enabled: the external path would fail", "warn")
    out.append(Check("settings in use", "ok", f"model {v.get('llm_model')}, top-k {v.get('retrieval_top_k')}, minimum retrieval score "
                     f"{v.get('retrieval_min_score')}, max rewrite retries {v.get('max_rewrite_retries')}"))
    return out


def check_dataset(path: str, n_cases: int, problems: Sequence[str]) -> Check:
    """The dataset loaded and every label was found in the knowledge-base documents."""
    if problems:
        return Check("dataset valid", "fail", f"{len(problems)} problem(s) in {path}: " + "; ".join(problems[:3]))
    return Check("dataset valid", "ok", f"{n_cases} cases in {path}; every label is found in the knowledge-base documents")


def check_fingerprint(actual: str | None, expected: str | None) -> Check:
    """A frozen dataset (the fresh block) still has the labels it was frozen with."""
    if expected is None:
        return Check("dataset fingerprint", "skip", "no .sha256 file next to the dataset (only the fresh block has one)")
    if actual != expected:
        return Check("dataset fingerprint", "fail", "the cases differ from the frozen fingerprint: labels were edited after freezing (record it in docs/FRESH_QUESTIONS.md)")
    return Check("dataset fingerprint", "ok", f"matches the frozen fingerprint {expected[:12]}...")


def check_split(planned: str, split_exists: bool, problems: Sequence[str], dataset_ids: Iterable[str], split_ids: Iterable[str]) -> Check:
    """The frozen dev/test split fits the dataset and the planned ``--split``."""
    ds, sp = set(dataset_ids), set(split_ids)
    if not split_exists:
        if planned != "all":
            return Check("split", "fail", f"--split {planned} needs the split file, which does not exist")
        return Check("split", "ok", "no split file; the whole dataset runs")
    if not ds & sp:
        status: Status = "fail" if planned != "all" else "warn"
        return Check("split", status, "a split file exists but covers none of this dataset's cases; for the fresh block point EVAL_SPLIT_PATH at a file that does not exist (docs/RUN_PLAN.md, Day 1)")
    if problems:
        return Check("split", "fail", f"{len(problems)} problem(s): " + "; ".join(problems[:3]))
    return Check("split", "ok", f"the split file matches the dataset ({len(ds & sp)} of {len(ds)} cases assigned); planned --split {planned}")


def check_documents(rows: Sequence[DocumentRow], core_names: Iterable[str]) -> list[Check]:
    """The SQLite ``documents`` table holds exactly the core knowledge-base files, all indexed."""
    core = set(core_names)
    names = {r[0] for r in rows}
    missing, extra = sorted(core - names), sorted(names - core)
    out = []
    if missing or extra:
        parts = ([f"missing: {', '.join(missing)}"] if missing else []) + ([f"extra: {', '.join(extra)} (labels refer to the core documents only; delete extras in Admin > Documents)"] if extra else [])
        out.append(Check("SQLite document set", "fail", "; ".join(parts)))
    else:
        out.append(Check("SQLite document set", "ok", f"exactly the {len(core)} core documents"))
    not_indexed = sorted(f"{r[0]} ({r[1]})" for r in rows if r[1] != "indexed")
    if not_indexed:
        out.append(Check("SQLite indexing status", "fail", "not indexed: " + ", ".join(not_indexed)))
    else:
        out.append(Check("SQLite indexing status", "ok", f"all {len(rows)} documents are indexed ({sum(1 for r in rows if r[2] == 'superseded')} marked superseded)"))
    return out


def check_index(index_chunks: Mapping[str, int] | None, error: str | None, rows: Sequence[DocumentRow], core_names: Iterable[str]) -> list[Check]:
    """The Pinecone namespace holds the same documents and chunk counts as SQLite (``index_chunks``: document name to chunk count)."""
    if error:
        return [Check("Pinecone index", "fail", f"could not read the index ({error})")]
    if index_chunks is None:
        return [Check("Pinecone index", "skip", "not checked (--skip-index); the run summary prints 'documents currently indexed' only afterwards")]
    core, names = set(core_names), set(index_chunks)
    missing, extra = sorted(core - names), sorted(names - core)
    out = []
    if missing or extra:
        parts = ([f"missing from the index: {', '.join(missing)}"] if missing else []) + ([f"extra in the index: {', '.join(extra)}"] if extra else [])
        out.append(Check("Pinecone document set", "fail", "; ".join(parts)))
    else:
        out.append(Check("Pinecone document set", "ok", f"exactly the {len(core)} core documents, {sum(index_chunks.values())} chunks"))
    recorded = {r[0]: r[3] for r in rows}
    off = sorted(f"{n} (index {index_chunks[n]}, SQLite {recorded[n]})" for n in index_chunks if n in recorded and index_chunks[n] != recorded[n])
    if off:
        out.append(Check("Pinecone chunk counts", "fail", "differ from SQLite (stale or duplicate vectors; re-index): " + ", ".join(off)))
    else:
        out.append(Check("Pinecone chunk counts", "ok", "match the chunk counts recorded in SQLite"))
    return out


def compare_settings(saved: Mapping[str, Any], current: Mapping[str, Any], *, label: str, consequence: str) -> Check:
    """Today's recorded settings against a run's saved ones. A key the older run never recorded cannot be compared."""
    keys = [k for k in saved if k in current]
    diff = sorted(k for k in keys if saved[k] != current[k])
    material = [k for k in diff if k not in STATISTICS_ONLY_SETTINGS]
    if material:
        shown = ", ".join(f"{k} {saved[k]!r} -> {current[k]!r}" for k in material)
        return Check(label, "fail", f"settings differ from the saved run ({shown}); {consequence}")
    if diff:
        return Check(label, "warn", "only interval settings differ (" + ", ".join(diff) + "); answers are unaffected, intervals will not match the earlier part")
    return Check(label, "ok", f"all {len(keys)} comparable settings are identical")


def check_resumable(run_id: str, exists: bool, status: str | None) -> Check:
    """A run named by ``--resume`` has a saved configuration and is not finished."""
    if not exists:
        return Check("resume target", "fail", f"run '{run_id}' has no saved configuration, so it cannot be resumed")
    if status == "completed":
        return Check("resume target", "fail", f"run '{run_id}' is already completed; there is nothing to resume")
    return Check("resume target", "ok", f"run '{run_id}' is {status or 'started without a summary'} and can be resumed")


# ------------------------------------------------------------------------------ reporting
def render(checks: Sequence[Check]) -> str:
    """One line per check."""
    return "\n".join(f"{TAGS[c.status]} {c.name}: {c.detail}" for c in checks)


def verdict(checks: Sequence[Check]) -> tuple[bool, str]:
    """``(ready, sentence)``: ready unless any check failed."""
    n = Counter(c.status for c in checks)
    if n["fail"]:
        return False, f"NOT READY: {n['fail']} check(s) failed. Fix them before spending tokens."
    return True, f"READY: {n['ok']} check(s) passed, {n['warn']} warning(s), {n['skip']} skipped."
