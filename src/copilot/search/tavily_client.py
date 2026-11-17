"""Tavily public web search with URL validation, bounded retries and official-source preference."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from copilot.config import Settings, get_settings
from copilot.logging_config import get_logger, redact
from copilot.retry import retry_call
from copilot.search.urls import domain_of, is_preferred_domain, validate_public_url

log = get_logger(__name__)


class ExternalSearchError(RuntimeError):
    """The search service failed after bounded retries (message is redacted)."""


class ExternalResult(BaseModel):
    """One validated public search result."""

    title: str
    url: str
    content: str
    domain: str
    preferred: bool = False  # official vendor documentation / status domain
    score: float | None = None
    published_date: str | None = None


class TavilySearchClient:
    """Searches PUBLIC sources only. Callers must pass an already-sanitised query."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self.call_count = 0  # number of Tavily API calls made (for audit/cost tracking)

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._settings.require_secret("tavily"))
        return self._client

    def _call(self, query: str, include_domains: list[str] | None) -> list[dict[str, Any]]:
        s = self._settings
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "query": query,
            "search_depth": s.tavily_search_depth,
            "max_results": s.tavily_max_results,
            "timeout": s.tavily_timeout_s,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        self.call_count += 1
        try:
            response = retry_call(
                lambda: client.search(**kwargs),
                operation="tavily.search",
                max_retries=s.service_max_retries,
                base_delay_s=s.service_retry_base_delay_s,
            )
        except Exception as exc:  # noqa: BLE001 - SDK/HTTP errors become one clear error
            raise ExternalSearchError(f"{type(exc).__name__}: {redact(str(exc))[:200]}") from exc
        return list((response or {}).get("results", []))

    def search(self, query: str) -> list[ExternalResult]:
        """Search official domains first; broaden to the general web unless two OFFICIAL results were found.

        Only genuinely official results count toward that quota: a community thread hosted on an official
        domain (e.g. learn.microsoft.com/answers/...) is user-written content and must not stop the broader
        search. At most two Tavily calls per search. Invalid URLs and duplicates are dropped; official
        sources are ranked first.
        """
        s = self._settings
        raw = self._call(query, s.preferred_external_domains) if s.preferred_external_domains else []
        official = sum(
            1 for item in raw
            if is_preferred_domain(str(item.get("url", "")), s.preferred_external_domains, s.user_generated_url_markers)
        )
        if official < 2:
            raw += self._call(query, None)

        results: dict[str, ExternalResult] = {}
        for item in raw:
            url = validate_public_url(str(item.get("url", "")))
            if not url or url in results:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            results[url] = ExternalResult(
                title=str(item.get("title") or domain_of(url))[:200],
                url=url,
                content=content[: s.external_content_max_chars],
                domain=domain_of(url),
                preferred=is_preferred_domain(url, s.preferred_external_domains, s.user_generated_url_markers),
                score=item.get("score"),
                published_date=item.get("published_date") or None,
            )
        ranked = sorted(results.values(), key=lambda r: (not r.preferred, -(r.score or 0.0)))
        log.info("External search done", extra={"results": len(ranked), "official": sum(r.preferred for r in ranked)})
        return ranked[: s.tavily_max_results]
