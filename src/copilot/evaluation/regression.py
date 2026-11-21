"""Retrieval regression check: re-measure retrieval on the labelled questions and compare it with a committed baseline.

A change to chunking, the text cleaner, the embedding model or the retriever can quietly make retrieval worse, and the
answer-level tests (which use fake embeddings) cannot see that. This check uses the REAL embedding model over the
synthetic KB, in an in-memory index (no Pinecone, no LLM, no API tokens), so it can run in CI.

What it is, and what it is not:

* It catches a drop in document recall, MRR and nDCG relative to the numbers the code produced when the baseline was
  last refreshed. The baseline is a snapshot, not a claim that retrieval is good.
* It measures ranking for the ORIGINAL question only (no query rewriting, no reranker unless the settings enable one).
* Refreshing the baseline is a deliberate act (``--update-retrieval-baseline``); a changed setup (model, chunking, KB
  files) refuses to compare against old numbers rather than guessing.
* Differences smaller than the tolerance are ignored: the same model on another machine can reorder near-ties.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from copilot.config import Settings
from copilot.evaluation.retrieval_metrics import rank_case
from copilot.evaluation.schemas import EvalCase

VERSION = 1
NOTE = (
    "Snapshot of retrieval quality measured with the real embedding model over the synthetic KB, used only to detect regressions. "
    "Refresh it deliberately (pytest tests/retrieval --update-retrieval-baseline) and review the diff."
)
ROUND = 4


def compared_keys(ks: Sequence[int]) -> list[str]:
    """Per-question metrics that are compared (all of them are higher-is-better)."""
    return [f"doc_recall@{k}" for k in ks] + ["doc_mrr"] + [f"doc_ndcg@{k}" for k in ks] + ["chunk_mrr"] + [f"chunk_ndcg@{k}" for k in ks]


def measure(cases: Sequence[EvalCase], retriever: Any, *, pool: int, ks: Sequence[int]) -> dict[str, dict[str, float | None]]:
    """Retrieval scores per question, for every question that names expected documents."""
    return {c.id: rank_case(retriever, c, pool, list(ks)).scores for c in cases if c.expected_documents}


def context(settings: Settings, kb_files: Sequence[str]) -> dict[str, Any]:
    """Everything that makes two measurements comparable. A difference here is a changed setup, not a regression."""
    return {
        "embedding_model": settings.embedding_model_name, "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
        "retrieval_pool": settings.eval_retrieval_pool, "ks": list(settings.eval_ks), "rerank_enabled": settings.rerank_enabled,
        "max_chunks_per_doc": settings.retrieval_max_chunks_per_doc, "kb_files": sorted(kb_files),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


def aggregate(per_case: Mapping[str, Mapping[str, float | None]], ids: Sequence[str], keys: Sequence[str]) -> dict[str, float | None]:
    """Mean of each metric over the given questions."""
    return {k: _mean([per_case[i].get(k) for i in ids]) for k in keys}


def make_baseline(per_case: Mapping[str, Mapping[str, float | None]], ctx: Mapping[str, Any], *, created_at: str | None = None) -> dict[str, Any]:
    """The JSON document stored in ``data/eval/retrieval_baseline.json``."""
    keys = compared_keys(ctx["ks"])
    ids = sorted(per_case)
    metrics = {k: (None if v is None else round(v, ROUND)) for k, v in aggregate(per_case, ids, keys).items()}
    return {
        "version": VERSION, "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": NOTE,
        "context": dict(ctx), "n_cases": len(ids), "metrics": metrics,
        "per_case": {i: {k: (None if per_case[i].get(k) is None else round(per_case[i][k], ROUND)) for k in keys} for i in ids},
    }


def dumps(baseline: Mapping[str, Any]) -> str:
    """Stable text form (sorted question ids, two-space indent) so a refresh produces a reviewable diff."""
    return json.dumps(baseline, indent=2, ensure_ascii=False) + "\n"


def load_baseline(path: Path) -> dict[str, Any]:
    """Read the baseline; ``ValueError`` (with the command that creates it) when it is missing or malformed."""
    hint = "Create it with: python -m pytest tests/retrieval --update-retrieval-baseline"
    if not path.is_file():
        raise ValueError(f"No retrieval baseline at {path}. {hint}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"The retrieval baseline {path} is not valid JSON ({exc}). {hint}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("per_case"), dict) or not isinstance(data.get("context"), dict):
        raise ValueError(f"The retrieval baseline {path} has an unexpected shape. {hint}")
    return data


def compare(baseline: Mapping[str, Any], per_case: Mapping[str, Mapping[str, float | None]], ctx: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    """Compare a fresh measurement with the baseline over the questions both contain.

    ``ok`` is False when the setup changed, nothing is comparable, or a mean metric fell by more than ``tolerance``.
    ``lost`` / ``gained`` list questions whose top-K document recall went down / up (informational: a single
    question can flip on a near-tie).
    """
    report: dict[str, Any] = {"ok": True, "problems": [], "rows": [], "n_compared": 0, "new_cases": [], "dropped_cases": [], "lost": [], "gained": [], "tolerance": tolerance}
    problems: list[str] = report["problems"]
    if baseline.get("version") != VERSION:
        problems.append(f"Baseline format version {baseline.get('version')!r} is not {VERSION}; refresh the baseline.")
    base_ctx = baseline.get("context") or {}
    for key in sorted(set(base_ctx) | set(ctx)):
        if base_ctx.get(key) != ctx.get(key):
            problems.append(f"Setup changed: {key} was {base_ctx.get(key)!r}, now {ctx.get(key)!r}. The numbers are not comparable; if the change is intended, refresh the baseline.")
    base_cases = baseline.get("per_case") or {}
    common = sorted(set(base_cases) & set(per_case))
    report["new_cases"] = sorted(set(per_case) - set(base_cases))
    report["dropped_cases"] = sorted(set(base_cases) - set(per_case))
    report["n_compared"] = len(common)
    if not problems and not common:
        problems.append("No question appears in both the baseline and this measurement.")
    if not problems:
        ks = list(ctx["ks"])
        for key in compared_keys(ks):
            before, after = _mean([base_cases[i].get(key) for i in common]), _mean([per_case[i].get(key) for i in common])
            if before is None or after is None:
                continue
            delta = after - before
            status = "regressed" if delta < -tolerance else ("improved" if delta > tolerance else "same")
            report["rows"].append({"metric": key, "baseline": before, "now": after, "delta": delta, "status": status})
            if status == "regressed":
                problems.append(f"{key} fell from {before:.3f} to {after:.3f} over {len(common)} questions (tolerance {tolerance:.2f}).")
        top = f"doc_recall@{max(ks)}"
        report["lost"] = [i for i in common if (base_cases[i].get(top) or 0.0) > (per_case[i].get(top) or 0.0) + 1e-9]
        report["gained"] = [i for i in common if (per_case[i].get(top) or 0.0) > (base_cases[i].get(top) or 0.0) + 1e-9]
    report["ok"] = not problems
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Plain-text summary for a test failure message or a terminal."""
    lines = ["Retrieval regression check: " + ("OK" if report["ok"] else "FAILED"), f"questions compared: {report['n_compared']}, tolerance {report['tolerance']:.2f}"]
    lines += [f"  PROBLEM: {p}" for p in report["problems"]]
    lines += [f"  {r['metric']:<16} baseline {r['baseline']:.3f}  now {r['now']:.3f}  ({r['delta']:+.3f}) {r['status']}" for r in report["rows"]]
    if report["lost"]:
        lines.append("  questions that lost a document hit at the largest K: " + ", ".join(report["lost"]))
    if report["gained"]:
        lines.append("  questions that gained a document hit at the largest K: " + ", ".join(report["gained"]))
    if report["new_cases"]:
        lines.append(f"  {len(report['new_cases'])} question(s) not in the baseline (refresh it to include them): " + ", ".join(report["new_cases"][:10]))
    if report["dropped_cases"]:
        lines.append(f"  {len(report['dropped_cases'])} baseline question(s) no longer in the dataset: " + ", ".join(report["dropped_cases"][:10]))
    return "\n".join(lines)
