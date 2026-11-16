"""Node 9: persist the full request lifecycle to SQLite."""
from __future__ import annotations

import time

from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced
from copilot.graph.state import CopilotState


def make_audit_logger(deps: GraphDeps):
    """Build the ``audit_logger`` node. Its own trace event is not part of the stored trace."""

    @traced("audit_logger")
    def audit_logger(state: CopilotState) -> dict:
        elapsed_ms = (time.perf_counter() - state.get("started_perf", time.perf_counter())) * 1000
        deps.audit.record_state(dict(state), execution_time_ms=elapsed_ms)
        return {"_event": event(f"audit record written for request {state.get('request_id')}", "end", execution_time_ms=round(elapsed_ms, 1))}

    return audit_logger
