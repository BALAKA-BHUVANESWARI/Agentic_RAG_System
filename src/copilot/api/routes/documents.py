"""Document management endpoints. Listing is filtered by role; mutations are admin-only."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, Response, UploadFile, status

from copilot.api.schemas import AccessLevelName, DeleteResponse, DocumentListResponse, DocumentOut, IngestionOut, LifecycleName
from copilot.api.security import Principal, get_principal, require_admin
from copilot.api.services import Services, get_services
from copilot.auth.access import allowed_access_levels

router = APIRouter(tags=["documents"])
DocumentId = Path(pattern=r"^[a-f0-9]{64}$", description="64-character hex document id")


def _to_out(rec, admin: bool) -> DocumentOut:
    data = rec.model_dump(include=set(DocumentOut.model_fields) - {"uploaded_by", "error"})
    if admin:
        data.update(uploaded_by=rec.uploaded_by, error=rec.error)
    return DocumentOut(**data)


@router.post("/documents/upload", response_model=IngestionOut, summary="Upload and ingest a document (admin)")
def upload_document(
    response: Response,
    file: UploadFile = File(..., description="PDF, DOCX or TXT"),
    category: str | None = Form(default=None, max_length=64),
    department: str | None = Form(default=None, max_length=64),
    access_level: AccessLevelName | None = Form(default=None),
    lifecycle: LifecycleName | None = Form(default=None, description="current or superseded; omit to use the document header"),
    superseded_by: str | None = Form(default=None, max_length=120, description="Replacing document, used when superseded"),
    principal: Principal = Depends(require_admin),
    svc: Services = Depends(get_services),
) -> IngestionOut:
    """Validate and ingest one file. 201 = indexed, 200 = skipped (unchanged duplicate), 400 = rejected."""
    limit = svc.settings.max_upload_bytes
    data = file.file.read(limit + 1)  # never buffer more than limit+1 bytes
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {svc.settings.max_upload_mb} MB.")
    result = svc.pipeline.ingest_bytes(
        file.filename or "", data, uploaded_by=principal.user_id, category=category, department=department, access_level=access_level,
        lifecycle=lifecycle, superseded_by=superseded_by,
    )
    out = IngestionOut(**result.model_dump(include=set(IngestionOut.model_fields)))
    response.status_code = {"indexed": status.HTTP_201_CREATED, "skipped": status.HTTP_200_OK}.get(result.status, status.HTTP_400_BAD_REQUEST)
    return out


@router.get("/documents", response_model=DocumentListResponse, summary="List documents you are authorised to see")
def list_documents(
    status_filter: str | None = Query(default=None, alias="status", pattern="^(pending|processing|indexed|failed)$"),
    principal: Principal = Depends(get_principal),
    svc: Services = Depends(get_services),
) -> DocumentListResponse:
    """Only documents whose access level the caller's role may read are returned."""
    records = svc.repo.list(access_levels=allowed_access_levels(principal.role, svc.settings), status=status_filter)
    docs = [_to_out(r, principal.is_admin) for r in records]
    return DocumentListResponse(role=principal.role, count=len(docs), documents=docs)


@router.delete("/documents/{document_id}", response_model=DeleteResponse, summary="Delete a document and its vectors (admin)")
def delete_document(document_id: str = DocumentId, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)) -> DeleteResponse:
    """Remove vectors, SQLite rows and the stored source file."""
    if not svc.pipeline.delete_document(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return DeleteResponse(document_id=document_id, deleted=True)


@router.post("/documents/{document_id}/reindex", response_model=IngestionOut, summary="Re-embed and re-index a stored document (admin)")
def reindex_document(
    response: Response, document_id: str = DocumentId, principal: Principal = Depends(require_admin), svc: Services = Depends(get_services)
) -> IngestionOut:
    """Re-index from the stored source file."""
    result = svc.pipeline.reindex(document_id)
    if result.error == "not_found":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    response.status_code = status.HTTP_200_OK if result.status != "failed" else status.HTTP_400_BAD_REQUEST
    return IngestionOut(**result.model_dump(include=set(IngestionOut.model_fields)))
