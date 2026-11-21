"""CSV exports of a run (the JSON summary is the primary artefact; CSV is for spreadsheets)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from copilot.evaluation.results import CaseResult
from copilot.evaluation.schemas import EvalCase

CASE_COLUMNS = [
    "case_id", "category", "mode", "expected_route", "correct", "outcome", "point_coverage", "citation_correct",
    "abstained", "abstention_reason", "used_external", "external_triggered", "tavily_calls", "retry_count",
    "grade_sufficient", "grade_confidence", "validation_passed", "latency_ms", "prompt_tokens", "completion_tokens",
    "llm_calls", "faithfulness", "relevance", "error", "request_id",
]


def case_rows(cases: list[EvalCase], results: list[CaseResult]) -> list[dict[str, Any]]:
    """One flat row per (case, mode)."""
    by_id = {c.id: c for c in cases}
    rows = []
    for r in sorted(results, key=lambda x: (x.case_id, x.mode)):
        c = by_id.get(r.case_id)
        rows.append({
            "case_id": r.case_id, "category": r.category, "mode": r.mode, "expected_route": c.expected_route if c else "",
            "correct": r.scores.get("correct"), "outcome": r.scores.get("outcome"), "point_coverage": r.scores.get("point_coverage"),
            "citation_correct": r.scores.get("citation_correct"), "abstained": r.abstained, "abstention_reason": r.abstention_reason,
            "used_external": r.used_external, "external_triggered": r.external_triggered, "tavily_calls": r.tavily_calls,
            "retry_count": r.retry_count, "grade_sufficient": r.grade_sufficient, "grade_confidence": r.grade_confidence,
            "validation_passed": r.validation_passed, "latency_ms": round(r.latency_ms, 1), "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "llm_calls": r.llm_calls,
            "faithfulness": r.judge.faithfulness if r.judge else None, "relevance": r.judge.relevance if r.judge else None,
            "error": r.error, "request_id": r.request_id,
        })
    return rows


def flatten_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows ``scope, metric, value, num, den`` covering retrieval, both modes and the comparison."""
    rows: list[dict[str, Any]] = []

    def add(scope: str, metric: str, v: Any) -> None:
        if isinstance(v, dict) and "value" in v:
            rows.append({"scope": scope, "metric": metric, "value": v["value"], "num": v.get("num"), "den": v.get("den")})
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            rows.append({"scope": scope, "metric": metric, "value": v, "num": None, "den": None})

    retrieval = summary.get("retrieval") or {}
    for k, v in (retrieval.get("metrics") or {}).items():
        add("retrieval", k, v)
    for mode, ms in (summary.get("modes") or {}).items():
        for section in ("answer", "routing", "system"):
            for k, v in (ms.get(section) or {}).items():
                add(f"{mode}.{section}", k, v)
        for cat, v in (ms.get("by_category") or {}).items():
            add(f"{mode}.by_category", cat, v)
    for row in (summary.get("comparison") or {}).get("rows", []):
        rows.append({"scope": "comparison", "metric": row["metric"], "value": row["verdict"], "num": row.get("baseline"), "den": row.get("agentic")})
    for mode, a in ((summary.get("attacks") or {}).get("modes") or {}).items():  # num = lower bound, den = upper bound (ASR row only)
        rows.append({"scope": f"attacks.{mode}", "metric": "attack_success_rate", "value": a["asr"]["value"], "num": a["asr"]["lo"], "den": a["asr"]["hi"]})
        rows.extend({"scope": f"attacks.{mode}", "metric": f"flag.{flag}", "value": n, "num": None, "den": None} for flag, n in a["by_flag"].items())
    iv = summary.get("intervals") or {}
    for mode, rates in (iv.get("rates") or {}).items():  # num = lower bound, den = upper bound
        for r in rates.values():
            rows.append({"scope": f"interval.{mode}", "metric": r["label"], "value": r["value"], "num": r["lo"], "den": r["hi"]})
    for p in iv.get("paired", []):  # value = difference (after - before); num = lower bound, den = upper bound
        for m in p["metrics"]:
            rows.append({"scope": f"paired.{p['from']}->{p['to']}", "metric": f"{m['label']} ({m['verdict']})", "value": m["diff"], "num": m["lo"], "den": m["hi"]})
    for step in (summary.get("ablation") or {}).get("steps", []):  # num = metric before the step, den = after it
        for c in step["changes"]:
            rows.append({"scope": f"ablation.{step['from']}->{step['to']}", "metric": c["metric"], "value": c["verdict"], "num": c["before"], "den": c["after"]})
    return rows


def _write(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_cases_csv(path: Path, cases: list[EvalCase], results: list[CaseResult]) -> None:
    """Write ``cases.csv``."""
    _write(path, CASE_COLUMNS, case_rows(cases, results))


def write_metrics_csv(path: Path, summary: dict[str, Any]) -> None:
    """Write ``metrics.csv`` (for the comparison scope: ``num`` = baseline value, ``den`` = agentic value)."""
    _write(path, ["scope", "metric", "value", "num", "den"], flatten_metrics(summary))
