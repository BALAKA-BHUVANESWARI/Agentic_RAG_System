"""Streamlit entry point.  Run:  streamlit run frontend/streamlit_app.py

The UI only calls the FastAPI backend (start it first with ``python scripts/run_api.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from copilot.config import get_settings  # noqa: E402
from copilot.ui import runtime, theme  # noqa: E402
from copilot.ui.views.admin_view import render_admin  # noqa: E402
from copilot.ui.views.chat_view import render_chat  # noqa: E402
from copilot.ui.views.sidebar import render_sidebar  # noqa: E402

st.set_page_config(page_title="IT Support Copilot", page_icon="🛠️", layout="wide")
theme.inject_theme()
settings = get_settings()
identity = render_sidebar(settings)
client = runtime.make_client(identity.role, identity.user_id)

if identity.page == "Admin" and identity.role == "admin":
    render_admin(client, settings)
else:
    render_chat(client, identity.allow_external)
