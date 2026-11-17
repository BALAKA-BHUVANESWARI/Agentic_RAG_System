"""Evaluation endpoints (admin only). Runs execute in the background; poll the status endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from copilot.api.schemas import EvaluationRunInfo, EvaluationRunRequest
from copilot.api.security import Principal, require_admin
from copilot.api.services import Services, get_services
from copilot.evaluation.runner import RunOptions
from copilot.evaluation.service import EvaluationBusyError, EvaluationService

router = APIRouter(tags=["evaluation"])


def _service(svc: Services) -> EvaluationService:
    if svc.evaluation_runner is None:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "The evaluation suite is not installed.")
    return svc.evaluation_runner


@router.post("/evaluation/run", response_model=EvaluationRunInfo, status_code=status.HTTP_202_ACCEPTED, summary="Start an evaluation run (admin)")
def run_evaluation(body: EvaluationRunRequest, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)) -> dict[str, Any]:
    """Start a run in the background and return its id. Only one run can be active at a time (409 otherwise)."""
    service = _service(svc)
    opts = RunOptions(
        mode=body.mode, limit=body.limit, per_category=body.per_category, categories=body.categories, use_judge=body.use_judge,
        run_id=body.resume_run_id, resume=body.resume_run_id is not None,
    )
    try:
        return service.start(opts, started_by=principal.user_id)
    except EvaluationBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/evaluation/runs", response_model=list[EvaluationRunInfo], summary="List evaluation runs (admin)")
def list_runs(limit: int = 20, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    """Most recent runs first (limit 1-100)."""
    return _service(svc).list_runs(max(1, min(limit, 100)))


@router.get("/evaluation/runs/{run_id}", response_model=EvaluationRunInfo, summary="Status and progress of a run (admin)")
def run_status(run_id: str, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)) -> dict[str, Any]:
    """Status, progress and headline metrics of one run."""
    info = _service(svc).status(run_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Evaluation run not found.")
    return info


@router.get("/evaluation/runs/{run_id}/summary", summary="Full metrics summary of a run (admin)")
def run_summary(run_id: str, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)) -> dict[str, Any]:
    """The run's ``summary.json`` (retrieval, answer, routing and system metrics, comparison, failed cases)."""
    summary = _service(svc).summary(run_id)
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No summary for this run yet.")
    return summary


@router.post("/evaluation/runs/{run_id}/cancel", summary="Cancel the active run after its current case (admin)")
def cancel_run(run_id: str, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)) -> dict[str, Any]:
    """Request cancellation. Finished cases are kept and the run can be resumed."""
    if not _service(svc).cancel(run_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "That run is not currently running.")
    return {"run_id": run_id, "cancel_requested": True}
