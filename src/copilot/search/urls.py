"""URL validation helpers for external sources."""
from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

_MAX_URL_LENGTH = 2048


def validate_public_url(url: str) -> str | None:
    """Return the cleaned URL if it is a well-formed public http(s) URL, else None.

    Rejects non-http(s) schemes, embedded credentials, localhost, private/loopback IPs,
    single-label hosts and over-long URLs.
    """
    if not url or len(url) > _MAX_URL_LENGTH:
        return None
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    if parsed.username or parsed.password:
        return None
    if host == "localhost" or "." not in host:
        return None
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return None
    except ValueError:
        pass  # a normal hostname
    return parsed.geturl()


def domain_of(url: str) -> str:
    """Lower-case host without a leading ``www.``."""
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_user_generated_path(url: str, markers: list[str]) -> bool:
    """True if the URL path looks like community content (Q&A, forum threads) rather than vendor docs."""
    path = (urlparse(url).path or "").lower() + "/"
    return any(m.lower() in path for m in markers if m)


def is_preferred_domain(url: str, preferred: list[str], user_generated_markers: list[str] | None = None) -> bool:
    """True if the URL is on a preferred (official) domain AND is not user-generated content.

    A community thread hosted on an official domain is still user-written, so it is not "official".
    """
    host = domain_of(url)
    on_domain = any(host == d.lower() or host.endswith("." + d.lower()) for d in preferred)
    return on_domain and not is_user_generated_path(url, user_generated_markers or [])
