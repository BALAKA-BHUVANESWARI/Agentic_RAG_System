"""HTTP client for the FastAPI backend, used by the Streamlit UI.

The UI never imports the agent, Pinecone or Groq; it only calls the API, so no API keys exist in
the frontend process path. The role/user headers are the DEMO identity (see ``api/security.py``).
"""
from __future__ import annotations

from typing import Any

import httpx


class ApiError(Exception):
    """A readable API failure (HTTP error, validation error or unreachable backend)."""

    def __init__(self, status_code: int, detail: str, request_id: str | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}"
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(f"{'.'.join(str(x) for x in e.get('loc', [])[1:])}: {e.get('msg', '')}".strip(": ") for e in detail)
    return str(detail)


class CopilotClient:
    """Thin typed wrapper over the API. Pass ``http`` to inject a client (tests use the ASGI TestClient)."""

    def __init__(self, base_url: str, role: str, user_id: str, *, http: httpx.Client | None = None, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.role = role
        self.user_id = user_id
        self._timeout = timeout
        self._injected = http is not None  # injected clients (tests) manage their own timeouts
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    @property
    def headers(self) -> dict[str, str]:
        """DEMO identity headers."""
        return {"X-Demo-Role": self.role, "X-Demo-User": self.user_id}

    def _request(self, method: str, path: str, *, accept: tuple[int, ...] = (), timeout: float | None = None, **kwargs: Any) -> Any:
        try:
            if not self._injected:
                kwargs["timeout"] = timeout or self._timeout
            resp = self._http.request(method, path, headers=self.headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(0, f"Cannot reach the API at {self.base_url} ({type(exc).__name__}). Is `python scripts/run_api.py` running?") from exc
        if resp.status_code >= 400 and resp.status_code not in accept:
            raise ApiError(resp.status_code, _detail(resp), resp.headers.get("X-Request-ID"))
        return resp.json()

    # ---- endpoints ----------------------------------------------------------------------
    def health(self) -> dict:
        """GET /health."""
        return self._request("GET", "/health", timeout=30)

    def chat(self, question: str, *, session_id: str | None = None, external_search_allowed: bool = True) -> dict:
        """POST /chat."""
        return self._request("POST", "/chat", json={"question": question, "session_id": session_id, "external_search_allowed": external_search_allowed})

    def list_documents(self, status: str | None = None) -> dict:
        """GET /documents."""
        return self._request("GET", "/documents", params={"status": status} if status else None)

    def upload_document(
        self, filename: str, data: bytes, *, category: str | None = None, department: str | None = None,
        access_level: str | None = None, lifecycle: str | None = None, superseded_by: str | None = None,
    ) -> dict:
        """POST /documents/upload. A rejected file returns its IngestionOut body (status 'failed') instead of raising.

        ``lifecycle`` is ``current`` or ``superseded``; leave it out to use the document header.
        """
        fields = {"category": category, "department": department, "access_level": access_level, "lifecycle": lifecycle, "superseded_by": superseded_by}
        form = {k: v for k, v in fields.items() if v}
        return self._request("POST", "/documents/upload", accept=(400,), files={"file": (filename, data)}, data=form)

    def delete_document(self, document_id: str) -> dict:
        """DELETE /documents/{id}."""
        return self._request("DELETE", f"/documents/{document_id}")

    def reindex_document(self, document_id: str) -> dict:
        """POST /documents/{id}/reindex (a failed re-index returns its body)."""
        return self._request("POST", f"/documents/{document_id}/reindex", accept=(400,))

    def feedback(self, request_id: str, rating: str, comment: str | None = None) -> dict:
        """POST /feedback."""
        return self._request("POST", "/feedback", json={"request_id": request_id, "rating": rating, "comment": comment})

    def audit(self, request_id: str) -> dict:
        """GET /audit/{request_id}."""
        return self._request("GET", f"/audit/{request_id}")

    # ---- evaluation (admin only; runs execute in the background on the server) --------------
    def start_evaluation(
        self, *, mode: str = "both", limit: int | None = None, per_category: int | None = None,
        categories: list[str] | None = None, use_judge: bool = True, resume_run_id: str | None = None,
        split: str = "all", attacks_only: bool = False, variants: list[str] | None = None,
    ) -> dict:
        """POST /evaluation/run. Returns at once with the run's status; poll :meth:`evaluation_status`."""
        body = {
            "mode": mode, "limit": limit, "per_category": per_category, "categories": categories, "use_judge": use_judge,
            "resume_run_id": resume_run_id, "split": split, "attacks_only": attacks_only, "variants": variants,
        }
        return self._request("POST", "/evaluation/run", json=body)

    def list_evaluation_runs(self, limit: int = 20) -> list[dict]:
        """GET /evaluation/runs (newest first)."""
        return self._request("GET", "/evaluation/runs", params={"limit": limit})

    def evaluation_status(self, run_id: str) -> dict:
        """GET /evaluation/runs/{id}."""
        return self._request("GET", f"/evaluation/runs/{run_id}")

    def evaluation_summary(self, run_id: str) -> dict:
        """GET /evaluation/runs/{id}/summary (404 until the run has written one)."""
        return self._request("GET", f"/evaluation/runs/{run_id}/summary")

    def cancel_evaluation(self, run_id: str) -> dict:
        """POST /evaluation/runs/{id}/cancel (the run stops after the case in flight)."""
        return self._request("POST", f"/evaluation/runs/{run_id}/cancel")
