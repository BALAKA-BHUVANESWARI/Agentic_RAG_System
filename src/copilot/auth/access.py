"""Local DEMO role-based access helpers.

LIMITATION: this is a demonstration mechanism (role passed by header or Streamlit
selector). It is NOT authentication and NOT production identity management: anyone
who can reach the API can claim any role. Real deployments need SSO/OIDC.
"""
from __future__ import annotations

from copilot.config import Settings, get_settings


def resolve_role(role: str | None, settings: Settings | None = None) -> str:
    """Return a known role, falling back to the configured default for None/empty."""
    settings = settings or get_settings()
    return (role or settings.default_role).strip().lower()


def allowed_access_levels(role: str | None, settings: Settings | None = None) -> list[str]:
    """Access levels the role may read. Unknown roles get an empty list (deny by default)."""
    settings = settings or get_settings()
    return list(settings.access_levels_for(resolve_role(role, settings)))


def can_read(role: str | None, access_level: str, settings: Settings | None = None) -> bool:
    """True if the role may read documents at ``access_level``."""
    return access_level in allowed_access_levels(role, settings)


def is_admin(role: str | None, settings: Settings | None = None) -> bool:
    """True only for the admin role."""
    return resolve_role(role, settings) == "admin"
