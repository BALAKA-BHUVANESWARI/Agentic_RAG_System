"""Unit tests for the query sanitizer, URL validation and the Tavily client (fake SDK)."""
from __future__ import annotations

import pytest

from copilot.search.sanitizer import find_sensitive, has_blocking_secret, redact_sensitive, sanitize_query
from copilot.search.tavily_client import ExternalSearchError, TavilySearchClient
from copilot.search.urls import domain_of, is_preferred_domain, validate_public_url

TERMS = ["Acme Demo Corp", "Acme", "SecureLink"]


def test_clean_public_question_passes_unchanged():
    r = sanitize_query("What is the latest Microsoft Teams outage guidance?", TERMS)
    assert r.query == "What is the latest Microsoft Teams outage guidance?" and r.usable and not r.blocked and r.removed == []


def test_identifiers_and_private_terms_are_stripped():
    q = "Is Acme SecureLink at vpn.acme-demo.example (10.1.2.3) down? ticket INC123456 from jdoe@acme.com"
    r = sanitize_query(q, TERMS)
    for leaked in ("acme", "SecureLink", "10.1.2.3", "INC123456", "jdoe", ".example"):
        assert leaked.lower() not in r.query.lower()
    assert {"email", "ip_address", "hostname", "ticket_id", "private_term"} <= set(r.removed)
    assert "()" not in r.query and "( )" not in r.query


@pytest.mark.parametrize("text", [
    "my password is hunter2 why does Teams fail",
    "token: abc123 breaks login",
    "use key gsk_abcdefghij1234567 for it",
    "Authorization: Bearer abcdefghij1234567890",
    "-----BEGIN RSA PRIVATE KEY----- MIIE",
])
def test_secrets_block_external_search(text):
    r = sanitize_query(text, TERMS)
    assert r.blocked and r.query == "" and not r.usable and has_blocking_secret(text)


def test_password_policy_question_is_not_a_secret():
    assert not has_blocking_secret("What is the password policy for VPN?")


def test_query_with_nothing_generic_left_is_unusable():
    r = sanitize_query("jdoe@acme.com 10.0.0.5 INC123456", TERMS)
    assert not r.blocked and not r.usable


def test_query_is_truncated_at_a_word_boundary():
    r = sanitize_query("outage guidance " * 40, [], max_chars=50)
    assert len(r.query) <= 50 and not r.query.endswith("guid")


def test_redact_sensitive_only_masks_secrets():
    assert redact_sensitive("login with password: hunter2 please") == "login with [REDACTED] please"
    assert redact_sensitive("email me at a@b.com") == "email me at a@b.com"
    assert [f.kind for f in find_sensitive("a@b.com")] == ["email"]


def test_url_validation():
    assert validate_public_url("https://learn.microsoft.com/a?b=1") == "https://learn.microsoft.com/a?b=1"
    for bad in ["ftp://a.com/x", "http://localhost/x", "http://127.0.0.1/x", "http://10.0.0.1/", "https://u:p@evil.com/",
                "https://intranet/x", "not a url", "", "https://" + "a" * 3000 + ".com"]:
        assert validate_public_url(bad) is None


def test_domain_helpers_resist_lookalikes():
    pref = ["learn.microsoft.com"]
    assert domain_of("https://www.Learn.Microsoft.com/x") == "learn.microsoft.com"
    assert is_preferred_domain("https://learn.microsoft.com/x", pref) and is_preferred_domain("https://a.learn.microsoft.com/x", pref)
    assert not is_preferred_domain("https://learn.microsoft.com.attacker.io/", pref)
    assert not is_preferred_domain("https://evillearn.microsoft.com/", pref)


# ---- Tavily client --------------------------------------------------------------------------
class _FakeTavily:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def _res(url, content="Some useful content", **kw):
    return {"title": kw.pop("title", "T"), "url": url, "content": content, **kw}


def test_official_first_then_general_search_when_few_results(make_settings):
    sdk = _FakeTavily(
        {"results": [_res("https://learn.microsoft.com/a", title="Official", score=0.5)]},
        {"results": [_res("https://blog.example.org/x", title="Blog", score=0.99), _res("https://learn.microsoft.com/a")]},
    )
    client = TavilySearchClient(make_settings(), client=sdk)
    out = client.search("teams outage guidance")
    assert [r.title for r in out] == ["Official", "Blog"]  # official ranked first despite lower score; deduped
    assert out[0].preferred and not out[1].preferred and out[0].domain == "learn.microsoft.com"
    assert "include_domains" in sdk.calls[0] and "include_domains" not in sdk.calls[1]
    assert client.call_count == 2 and sdk.calls[0]["max_results"] == 5 and sdk.calls[0]["timeout"] == 15


def test_no_second_call_when_official_results_suffice(make_settings):
    sdk = _FakeTavily({"results": [_res("https://learn.microsoft.com/a"), _res("https://support.microsoft.com/b")]})
    client = TavilySearchClient(make_settings(), client=sdk)
    assert len(client.search("q words here")) == 2 and client.call_count == 1


def test_invalid_urls_empty_content_are_dropped_and_content_truncated(make_settings):
    s = make_settings(external_content_max_chars=200)
    sdk = _FakeTavily(
        {"results": [_res("ftp://x.com/a"), _res("http://localhost/a"), _res("https://u:p@x.com/a"),
                     _res("https://ok.example.org/a", content=""), _res("https://ok.example.org/b", content="z" * 500, published_date="2026-11-01")]},
        {"results": []},
    )
    out = TavilySearchClient(s, client=sdk).search("q words here")
    assert [r.url for r in out] == ["https://ok.example.org/b"] and len(out[0].content) == 200 and out[0].published_date == "2026-11-01"


def test_failures_are_bounded_and_redacted(make_settings):
    s = make_settings(service_max_retries=1, service_retry_base_delay_s=0.0)
    sdk = _FakeTavily(*[ConnectionError("down api_key=abc123456789")] * 4)
    client = TavilySearchClient(s, client=sdk)
    with pytest.raises(ExternalSearchError) as exc:
        client.search("q words here")
    assert "abc123456789" not in str(exc.value) and len(sdk.calls) == 2


def test_missing_key_is_a_clear_error(make_settings, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        TavilySearchClient(make_settings()).search("q words here")


def test_community_content_on_official_domain_is_not_official():
    pref, marks = ["learn.microsoft.com", "techcommunity.microsoft.com"], ["/answers/", "/discussions/"]
    assert is_preferred_domain("https://learn.microsoft.com/en-us/microsoft-365/enterprise/view-service-health", pref, marks)
    assert not is_preferred_domain("https://learn.microsoft.com/en-au/answers/questions/6069277/microsoft-365-is-down", pref, marks)
    assert not is_preferred_domain("https://techcommunity.microsoft.com/discussions/microsoft-365/outage-updates/4573007", pref, marks)
    assert is_preferred_domain("https://learn.microsoft.com/answers-guide", pref, marks)
