"""SQLite persistence of evaluation run metadata (table ``eval_runs``).

The table has no status column, so the lifecycle status lives in the ``metadata`` JSON
(``running | completed | cancelled | stopped | failed``). Detailed results live in files under the run directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from copilot.db.connection import connect
from copilot.utils import utc_now_iso


def _row(r) -> dict[str, Any]:
    meta = json.loads(r["metadata"]) if r["metadata"] else {}
    return {
        "run_id": r["run_id"], "mode": r["mode"], "dataset_path": r["dataset_path"], "num_cases": r["num_cases"],
        "metrics": json.loads(r["metrics"]) if r["metrics"] else None, "results_path": r["results_path"],
        "metadata": meta, "status": meta.get("status", "unknown"), "started_at": r["started_at"], "finished_at": r["finished_at"],
    }


class EvalRunStore:
    """Create, finish and read run rows."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)

    def start_run(self, run_id: str, mode: str, dataset_path: str, num_cases: int, metadata: dict[str, Any]) -> None:
        """Insert the run (or mark an existing one as running again when resuming)."""
        meta = json.dumps({**metadata, "status": "running"})
        with connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO eval_runs(run_id, mode, dataset_path, num_cases, metadata, started_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET metadata = excluded.metadata, finished_at = NULL",
                (run_id, mode, dataset_path, num_cases, meta, utc_now_iso()),
            )

    def finish_run(self, run_id: str, *, status: str, metrics: dict[str, Any] | None, results_path: str, extra: dict[str, Any] | None = None) -> None:
        """Record the final status, headline metrics and where the result files are."""
        with connect(self._db_path) as conn:
            row = conn.execute("SELECT metadata FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()
            meta = json.loads(row["metadata"]) if row and row["metadata"] else {}
            meta.update({**(extra or {}), "status": status})
            conn.execute(
                "UPDATE eval_runs SET metrics = ?, results_path = ?, metadata = ?, finished_at = ? WHERE run_id = ?",
                (json.dumps(metrics) if metrics is not None else None, results_path, json.dumps(meta), utc_now_iso(), run_id),
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        """One run row, or None."""
        with connect(self._db_path) as conn:
            r = conn.execute("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row(r) if r else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent runs first."""
        with connect(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM eval_runs ORDER BY started_at DESC, run_id DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r) for r in rows]
