"""Local DEMO identity for the API.

LIMITATION (by design): the caller states its identity in two headers, ``X-Demo-User`` and
``X-Demo-Role``. Nothing proves the claim, so anyone who can reach the API can claim ``admin``.
This exists to demonstrate document-level authorization, NOT to provide authentication.
A real deployment must put SSO/OIDC in front and derive the principal from a verified token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from copilot.api.services import Services, get_services
from copilot.auth.access import is_admin, resolve_role

_USER_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


@dataclass(frozen=True)
class Principal:
    """Who is calling (as claimed by the demo headers)."""

    user_id: str
    role: str

    @property
    def is_admin(self) -> bool:
        """True for the admin role."""
        return self.role == "admin"


def get_principal(
    svc: Services = Depends(get_services),
    x_demo_role: str | None = Header(default=None, description="DEMO ONLY: employee | admin"),
    x_demo_user: str | None = Header(default=None, description="DEMO ONLY: caller identifier"),
) -> Principal:
    """Resolve and validate the demo identity. Unknown roles are denied (no silent downgrade)."""
    role = resolve_role(x_demo_role, svc.settings)
    if role not in svc.settings.role_access_levels:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Unknown role.")
    user = (x_demo_user or "anonymous").strip()
    if not _USER_RE.match(user):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "X-Demo-User must be 1-64 characters: letters, digits, _ . @ -")
    return Principal(user_id=user, role=role)


def require_admin(principal: Principal = Depends(get_principal), svc: Services = Depends(get_services)) -> Principal:
    """Dependency for admin-only endpoints."""
    if not is_admin(principal.role, svc.settings):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator role required.")
    return principal
