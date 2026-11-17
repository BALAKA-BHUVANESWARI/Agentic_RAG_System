"""Bounded retry with exponential backoff for calls to external services."""
from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

from copilot.logging_config import get_logger

T = TypeVar("T")
log = get_logger(__name__)


_HINT = re.compile(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)(ms|s)", re.IGNORECASE)


def server_wait_hint(exc: BaseException) -> float | None:
    """Seconds the service asked us to wait (Retry-After header or 'try again in 6.2s'), if any."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers else None
        if value:
            return float(value)
    except (TypeError, ValueError, AttributeError):
        pass
    m = _HINT.search(str(exc))
    if not m:
        return None
    minutes, number, unit = m.groups()
    seconds = float(number) / (1000 if unit.lower() == "ms" else 1)
    return seconds + (60 * int(minutes) if minutes else 0)


def retry_call(
    fn: Callable[[], T],
    *,
    operation: str,
    max_retries: int,
    base_delay_s: float,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    max_delay_s: float = 30.0,
) -> T:
    """Call ``fn`` and retry up to ``max_retries`` extra times with exponential backoff.

    The final exception is re-raised so callers can degrade gracefully.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on as exc:
            if attempt >= max_retries:
                log.error("%s failed after %d attempt(s): %s", operation, attempt + 1, type(exc).__name__)
                raise
            delay = base_delay_s * (2**attempt)
            hint = server_wait_hint(exc)
            if hint is not None:
                delay = max(delay, hint + 0.25)
            delay = min(delay, max_delay_s)
            # The service's own explanation (which limit, requested vs allowed) is what tells a
            # momentary rate limit from a request that can never fit; keep it short and secret-free.
            detail = " ".join(str(exc).split())[:300]
            log.warning("%s failed (%s); retrying in %.2fs: %s", operation, type(exc).__name__, delay, detail)
            time.sleep(delay)
            attempt += 1
