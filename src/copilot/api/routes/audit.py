"""GET /audit/{request_id}: authorised audit record retrieval."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status

from copilot.api.schemas import AuditResponse
from copilot.api.security import Principal, get_principal
from copilot.api.services import Services, get_services

router = APIRouter(tags=["audit"])


@router.get("/audit/{request_id}", response_model=AuditResponse, summary="Get the audit record of a request")
def get_audit(
    request_id: str = Path(pattern=r"^[A-Za-z0-9_-]{1,64}$"),
    principal: Principal = Depends(get_principal),
    svc: Services = Depends(get_services),
) -> AuditResponse:
    """Admins can read any record; others only their own. Everything else is 404 (existence is not revealed)."""
    record = svc.audit.get(request_id)
    if record is None or (not principal.is_admin and record.user_id != principal.user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audit record not found.")
    return AuditResponse(**record.model_dump(), feedback=svc.feedback.for_request(request_id))
