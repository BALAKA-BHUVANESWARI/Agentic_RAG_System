"""Background evaluation jobs for the API: one run at a time, started in a thread, observable and cancellable.

Runs can take many minutes (rate-limited LLM calls), so ``POST /evaluation/run`` starts a job and returns at once;
progress is read through :meth:`status`. A server restart interrupts a running job (it shows as ``interrupted``);
it can be resumed from its files.
"""
from __future__ import annotations

import threading
from typing import Any

from copilot.evaluation.runner import RUN_ID_RE, EvaluationRunner, RunOptions, new_run_id
from copilot.evaluation.store import EvalRunStore
from copilot.logging_config import get_logger, redact

log = get_logger(__name__)


class EvaluationBusyError(RuntimeError):
    """Another evaluation is already running."""


class EvaluationService:
    """Owns the single active job and answers status queries."""

    def __init__(self, runner: EvaluationRunner, store: EvalRunStore) -> None:
        self._runner, self._store = runner, store
        self._lock = threading.Lock()
        self._active: str | None = None
        self._progress: dict[str, Any] = {}
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- starting / cancelling -------------------------------------------------------
    def start(self, opts: RunOptions, *, started_by: str = "admin") -> dict[str, Any]:
        """Validate the request, start the job, return its initial status. Raises ``ValueError`` / ``EvaluationBusyError``."""
        opts, cases = self._runner.prepare(opts)  # fails fast (bad filters, bad resume id, dataset problems)
        run_id = opts.run_id if opts.resume else (opts.run_id or new_run_id())
        run_dir = self._runner.run_dir(run_id)  # rejects unsafe ids
        if not opts.resume and run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"Run '{run_id}' already exists; resume it or choose another id.")
        opts = opts.model_copy(update={"run_id": run_id})
        total = len(cases) * len(opts.modes)
        with self._lock:
            if self._active:
                raise EvaluationBusyError(f"Evaluation run '{self._active}' is still running.")
            self._active = run_id
            self._cancel = threading.Event()
            self._progress = {"done": 0, "total": total, "message": "starting"}
            cancel = self._cancel
            self._thread = threading.Thread(target=self._work, args=(opts, cancel), name=f"eval-{run_id}", daemon=True)
            self._thread.start()
        log.info("Evaluation started", extra={"run_id": run_id, "by": started_by})
        return {"run_id": run_id, "status": "running", "mode": opts.mode, "num_cases": len(cases), **self._progress_view()}

    def cancel(self, run_id: str) -> bool:
        """Ask the active run to stop after the current case. False if ``run_id`` is not running."""
        with self._lock:
            if self._active != run_id:
                return False
            self._cancel.set()
            self._progress["message"] = "cancelling after the current case"
            return True

    def active_run_id(self) -> str | None:
        """Id of the running job, if any."""
        with self._lock:
            return self._active

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current job ends (used by tests and the CLI)."""
        t = self._thread
        if t is not None:
            t.join(timeout)

    # ---- reading -----------------------------------------------------------------------
    def status(self, run_id: str) -> dict[str, Any] | None:
        """Status of one run (None if unknown or the id is invalid)."""
        if not RUN_ID_RE.match(run_id or ""):
            return None
        row = self._store.get(run_id)
        active = self.active_run_id() == run_id
        if row is None and not active:
            return None
        return self._info(row or {"run_id": run_id, "metadata": {}, "status": "running"}, active)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent runs, newest first."""
        active = self.active_run_id()
        rows = self._store.list(limit)
        out = [self._info(r, r["run_id"] == active) for r in rows]
        if active and not any(r["run_id"] == active for r in rows):
            # The job's database row is written by its worker thread; a request right after start can arrive first.
            out.insert(0, self._info({"run_id": active, "metadata": {}, "status": "running"}, True))
        return out

    def summary(self, run_id: str) -> dict[str, Any] | None:
        """``summary.json`` of a run (None until it exists)."""
        return self._runner.read_summary(run_id)

    # ---- internals -------------------------------------------------------------------------
    def _progress_view(self) -> dict[str, Any]:
        with self._lock:
            return {"done": self._progress.get("done", 0), "total": self._progress.get("total", 0), "message": self._progress.get("message", "")}

    def _info(self, row: dict[str, Any], active: bool) -> dict[str, Any]:
        meta = row.get("metadata") or {}
        status = row.get("status", "unknown")
        if active:
            status = "cancelling" if self._cancel.is_set() else "running"
            prog = self._progress_view()
        else:
            if status == "running":
                status = "interrupted"  # the process that was running it is gone
            disk = self._runner.read_progress(row["run_id"])
            prog = {"done": disk["done"], "total": disk["total"], "message": ""}
        return {
            "run_id": row["run_id"], "status": status, "mode": row.get("mode"), "dataset_path": row.get("dataset_path"),
            "num_cases": row.get("num_cases"), "use_judge": meta.get("use_judge"), "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"), "error": meta.get("error"), "metrics": row.get("metrics"), **prog,
        }

    def _on_progress(self, done: int, total: int, message: str) -> None:
        with self._lock:
            self._progress.update(done=done, total=total, message=message)

    def _work(self, opts: RunOptions, cancel: threading.Event) -> None:
        try:
            self._runner.run(opts, progress=self._on_progress, should_cancel=cancel.is_set)
        except Exception as exc:  # noqa: BLE001 - never let the thread die silently
            log.exception("Evaluation job crashed")
            try:
                self._store.finish_run(opts.run_id or "", status="failed", metrics=None, results_path="", extra={"error": redact(str(exc))[:300]})
            except Exception:  # noqa: BLE001
                pass
        finally:
            with self._lock:
                self._active = None
