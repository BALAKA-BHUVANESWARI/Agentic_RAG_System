"""POST /feedback: thumbs up/down on an answer you received."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from copilot.api.schemas import FeedbackRequest, FeedbackResponse
from copilot.api.security import Principal, get_principal
from copilot.api.services import Services, get_services

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED, summary="Rate an answer")
def post_feedback(body: FeedbackRequest, principal: Principal = Depends(get_principal), svc: Services = Depends(get_services)) -> FeedbackResponse:
    """Employees can rate only their own requests; admins can rate any. Unknown or foreign ids return 404."""
    record = svc.audit.get(body.request_id)
    if record is None or (not principal.is_admin and record.user_id != principal.user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found.")
    fid = svc.feedback.add(body.request_id, principal.user_id, body.rating, body.comment)
    return FeedbackResponse(feedback_id=fid, request_id=body.request_id, rating=body.rating)
