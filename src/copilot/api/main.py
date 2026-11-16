"""FastAPI application factory.

Run locally:  uvicorn copilot.api.main:create_app --factory --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from copilot.api.routes import ALL_ROUTERS
from copilot.api.services import Services, build_services
from copilot.config import Settings, get_settings
from copilot.logging_config import bind_request_id, get_logger, new_request_id, setup_logging

log = get_logger("copilot.api")

DESCRIPTION = (
    "Private-first Agentic RAG for IT support. **DEMO AUTH ONLY**: identity comes from the `X-Demo-User` and "
    "`X-Demo-Role` headers and is not verified; this is not production identity management."
)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the app. Tests pass ``services`` (with fakes); production builds real ones at startup."""
    settings = settings or (services.settings if services else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging()
        if services is None:
            app.state.services = build_services(settings)
            try:  # load the embedding model now so the first user request is not slow
                from copilot.embeddings.service import get_embedding_service

                get_embedding_service().embed_query("warm-up")
            except Exception as exc:  # noqa: BLE001 - /health still works; first request will retry
                log.warning("Embedding warm-up failed: %s", type(exc).__name__)
        yield

    app = FastAPI(title="IT Support Copilot API", version="0.1.0", description=DESCRIPTION, lifespan=lifespan)
    if services is not None:
        app.state.services = services

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001
        rid = new_request_id()
        request.state.request_id = rid
        started = time.perf_counter()
        with bind_request_id(rid):
            response = await call_next(request)
            log.info("request", extra={"method": request.method, "path": request.url.path, "status": response.status_code,
                                       "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        rid = getattr(request.state, "request_id", None)
        log.exception("Unhandled error")  # details go to the log only, never to the client
        return JSONResponse({"detail": "Internal server error.", "request_id": rid}, status_code=500, headers={"X-Request-ID": rid or ""})

    for router in ALL_ROUTERS:
        app.include_router(router)
    return app
