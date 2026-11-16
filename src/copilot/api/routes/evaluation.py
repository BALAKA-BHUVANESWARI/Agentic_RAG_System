"""POST /evaluation/run (admin). The evaluation suite is implemented in Phase 7 and plugs in via Services."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from copilot.api.schemas import EvaluationRunRequest
from copilot.api.security import Principal, require_admin
from copilot.api.services import Services, get_services

router = APIRouter(tags=["evaluation"])


@router.post("/evaluation/run", summary="Run the evaluation suite (admin)")
def run_evaluation(body: EvaluationRunRequest, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)) -> dict:
    """Run the labelled evaluation. Returns 501 until the evaluation module is installed."""
    if svc.evaluation_runner is None:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "The evaluation suite is not installed yet (Phase 7).")
    return svc.evaluation_runner.run(mode=body.mode, limit=body.limit, categories=body.categories)
