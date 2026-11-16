"""GET /health: application and dependency status. Reveals no secrets."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from copilot.api.schemas import DependencyStatus, HealthResponse
from copilot.api.services import Services, get_services
from copilot.db.connection import connect

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(svc: Services = Depends(get_services)) -> HealthResponse:
    """Cheap checks only: SQLite, Pinecone reachability and whether API keys are configured (no paid calls)."""
    s = svc.settings
    deps: list[DependencyStatus] = []
    try:
        with connect(s.sqlite_path) as conn:
            conn.execute("SELECT 1").fetchone()
        deps.append(DependencyStatus(name="sqlite", ok=True))
    except Exception as exc:  # noqa: BLE001
        deps.append(DependencyStatus(name="sqlite", ok=False, detail=type(exc).__name__))
    deps.append(DependencyStatus(name="pinecone", ok=bool(s.pinecone_api_key) and bool(svc.store.health_check()),
                                 detail="" if s.pinecone_api_key else "PINECONE_API_KEY not set"))
    deps.append(DependencyStatus(name="groq", ok=bool(s.groq_api_key), detail=f"model={s.groq_model}" if s.groq_api_key else "GROQ_API_KEY not set"))
    deps.append(DependencyStatus(name="tavily", ok=bool(s.tavily_api_key) or not s.external_search_enabled,
                                 detail="external search disabled" if not s.external_search_enabled else ("" if s.tavily_api_key else "TAVILY_API_KEY not set")))
    return HealthResponse(status="ok" if all(d.ok for d in deps) else "degraded", dependencies=deps)
