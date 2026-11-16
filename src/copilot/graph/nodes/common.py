"""Shared node helpers: the ``traced`` decorator and small utilities."""
from __future__ import annotations

import functools
import time
from typing import Any, Callable

from copilot.graph.state import CopilotState, TraceEvent
from copilot.llm.client import LLMUsage
from copilot.logging_config import bind_request_id, get_logger, redact
from copilot.utils import utc_now_iso

log = get_logger(__name__)
NodeFn = Callable[[CopilotState], dict[str, Any]]


def event(summary: str, decision: str | None = None, **data: Any) -> dict[str, Any]:
    """Describe what a node did; ``traced`` turns this into a real TraceEvent."""
    return {"summary": summary, "decision": decision, "data": data}


def usage_update(usage: LLMUsage | None) -> dict[str, int]:
    """LLM usage in the shape expected by the ``llm_usage`` reducer."""
    return usage.model_dump() if usage else {}


def traced(name: str) -> Callable[[NodeFn], NodeFn]:
    """Wrap a node: bind the request ID, time it, append a TraceEvent, and contain crashes.

    An unexpected exception becomes ``fatal_error`` (redacted); routing then sends the
    request to the abstention path instead of crashing the graph.
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        def wrapper(state: CopilotState) -> dict[str, Any]:
            started_at, t0 = utc_now_iso(), time.perf_counter()
            with bind_request_id(state.get("request_id")):
                try:
                    update = fn(state)
                except Exception as exc:  # noqa: BLE001 - last-resort containment
                    log.exception("Node crashed", extra={"node": name})
                    error = redact(f"{type(exc).__name__}: {exc}")[:300]
                    update = {"fatal_error": error, "_event": event(f"unexpected error: {type(exc).__name__}", "error")}
            ev = update.pop("_event", {}) or {}
            update["trace"] = [
                TraceEvent(
                    node=name, started_at=started_at, duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    summary=ev.get("summary", ""), decision=ev.get("decision"), data=ev.get("data", {}),
                ).model_dump()
            ]
            return update

        return wrapper

    return decorator
