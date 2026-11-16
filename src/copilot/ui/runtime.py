"""Builds the API client for the current demo identity (a seam that tests replace)."""
from __future__ import annotations

from copilot.config import get_settings
from copilot.ui.api_client import CopilotClient


def make_client(role: str, user_id: str) -> CopilotClient:
    """Create a client bound to the configured API URL and the chosen demo identity."""
    s = get_settings()
    return CopilotClient(s.api_base_url, role, user_id, timeout=s.ui_request_timeout_s)
