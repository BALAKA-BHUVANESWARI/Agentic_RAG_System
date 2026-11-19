"""Sidebar: demo identity, options, backend health."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from copilot.config import Settings
from copilot.ui import runtime, theme


@dataclass(frozen=True)
class Identity:
    """The demo identity and options chosen in the sidebar."""

    role: str
    user_id: str
    allow_external: bool
    page: str


def render_sidebar(settings: Settings) -> Identity:
    """Draw the sidebar and return the current selections."""
    with st.sidebar:
        st.markdown(theme.wordmark(), unsafe_allow_html=True)
        st.warning(
            "**Local demo identity.** You choose your own role here. This is NOT authentication or production "
            "identity management; the API trusts the role it is told.",
            icon="⚠️",
        )
        roles = sorted(settings.role_access_levels, key=lambda r: (r != settings.default_role, r))
        role = st.selectbox("Role", roles, key="role", help="employee: public + internal documents. admin: also restricted documents and management.")
        user_id = st.text_input("User ID", value="demo-user", key="user_id", max_chars=64)
        allow_external = st.checkbox("Allow public web search when needed", value=True, key="allow_external",
                                     help="Company documents are always searched first. Untick to forbid any public search.")
        pages = ["Chat", "Admin"] if role == "admin" else ["Chat"]
        page = st.radio("Page", pages, key="page")
        st.divider()
        if st.button("Clear conversation", key="clear"):
            st.session_state["messages"] = []
            st.session_state["feedback"] = {}
        if st.button("Check backend health", key="health"):
            try:
                h = runtime.make_client(role, user_id).health()
                (st.success if h["status"] == "ok" else st.warning)(f"Backend: {h['status']}")
                for d in h["dependencies"]:
                    st.caption(f"{'✅' if d['ok'] else '❌'} {d['name']} {d.get('detail') or ''}")
            except Exception as exc:  # noqa: BLE001 - show any failure, never crash the sidebar
                st.error(str(exc))
        st.caption(f"API: {settings.api_base_url}")
    return Identity(role=role, user_id=user_id.strip() or "demo-user", allow_external=allow_external, page=page)
