"""Unit tests for the Groq client (scripted fake SDK, no network)."""
from __future__ import annotations

import pytest
from fakes import FakeGroqSDK
from pydantic import BaseModel

from copilot.llm.client import GroqClient, LLMUnavailableError, StructuredOutputError, parse_json_object


class Out(BaseModel):
    x: int


@pytest.fixture
def make_client(make_settings):
    def _make(outcomes, retries=2):
        s = make_settings(groq_max_retries=retries, service_retry_base_delay_s=0.0)
        sdk = FakeGroqSDK(outcomes)
        return GroqClient(s, client=sdk, retry_exceptions=(ConnectionError,)), sdk

    return _make


def test_chat_returns_text_usage_and_uses_configured_model(make_client, make_settings):
    client, sdk = make_client(["hello"])
    r = client.chat([{"role": "user", "content": "hi"}])
    assert r.text == "hello" and r.usage.total_tokens == 15 and r.usage.calls == 1
    kwargs = sdk.calls[0]
    assert kwargs["model"] == make_settings().groq_model and kwargs["temperature"] == 0.0
    assert "response_format" not in kwargs
    assert client.usage_total.calls == 1


def test_transient_error_is_retried(make_client):
    client, sdk = make_client([ConnectionError("blip"), "ok"])
    assert client.chat([{"role": "user", "content": "hi"}]).text == "ok"
    assert len(sdk.calls) == 2 and client.usage_total.calls == 1


def test_retries_are_bounded_then_unavailable(make_client):
    client, sdk = make_client([ConnectionError("down")] * 3, retries=2)
    with pytest.raises(LLMUnavailableError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) == 3  # 1 attempt + 2 retries, never more


def test_non_retryable_error_fails_fast(make_client):
    client, sdk = make_client([ValueError("bad request")])
    with pytest.raises(LLMUnavailableError, match="ValueError"):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) == 1


def test_error_messages_are_redacted(make_client):
    client, _ = make_client([ValueError("failed api_key=abc123456789 for gsk_abcdefghij12345")])
    with pytest.raises(LLMUnavailableError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert "abc123456789" not in str(exc.value) and "gsk_abcdefghij" not in str(exc.value)


def test_missing_api_key_is_a_clear_config_error(make_settings, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqClient(make_settings()).chat([{"role": "user", "content": "hi"}])


def test_structured_success_uses_json_mode_and_schema_instruction(make_client):
    client, sdk = make_client(['{"x": 3}'])
    parsed, usage = client.structured([{"role": "user", "content": "q"}], Out)
    assert parsed == Out(x=3) and usage.calls == 1
    assert sdk.calls[0]["response_format"] == {"type": "json_object"}
    assert sdk.calls[0]["messages"][0]["role"] == "system" and "JSON Schema" in sdk.calls[0]["messages"][0]["content"]


def test_structured_accepts_code_fences(make_client):
    client, _ = make_client(['```json\n{"x": 4}\n```'])
    assert client.structured([{"role": "user", "content": "q"}], Out)[0].x == 4


def test_structured_repairs_once_and_aggregates_usage(make_client):
    client, sdk = make_client(["not json at all", '{"x": 5}'])
    parsed, usage = client.structured([{"role": "user", "content": "q"}], Out)
    assert parsed.x == 5 and usage.calls == 2 and usage.total_tokens == 30
    assert "not valid JSON" in sdk.calls[1]["messages"][-1]["content"]


def test_structured_gives_up_after_one_repair(make_client):
    client, sdk = make_client(["nope", '{"x": "abc"}'])
    with pytest.raises(StructuredOutputError):
        client.structured([{"role": "user", "content": "q"}], Out)
    assert len(sdk.calls) == 2


def test_parse_json_object():
    assert parse_json_object('Here you go: {"a": 1} thanks') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no braces")


def test_reasoning_effort_is_sent_via_extra_body_only_when_configured(make_settings):
    for effort, expected in (("low", {"reasoning_effort": "low"}), ("", None)):
        sdk = FakeGroqSDK(["ok"])
        s = make_settings(groq_reasoning_effort=effort)
        GroqClient(s, client=sdk, retry_exceptions=(ConnectionError,)).chat([{"role": "user", "content": "hi"}])
        assert sdk.calls[0].get("extra_body") == expected


def test_default_model_and_token_budget(make_settings):
    s = make_settings()
    assert s.groq_model == "openai/gpt-oss-120b" and s.llm_max_tokens >= 2048


def test_server_wait_hint_is_parsed_and_honoured(monkeypatch):
    from copilot import retry as r

    class Boom(Exception):
        pass

    assert r.server_wait_hint(Boom("Please try again in 6.5s. Need more")) == 6.5
    assert r.server_wait_hint(Boom("try again in 1m3.5s")) == 63.5
    assert r.server_wait_hint(Boom("try again in 250ms")) == 0.25
    assert r.server_wait_hint(Boom("nothing")) is None

    sleeps, calls = [], []
    monkeypatch.setattr(r.time, "sleep", sleeps.append)

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise Boom("try again in 6.5s")
        return "ok"

    assert r.retry_call(fn, operation="t", max_retries=2, base_delay_s=0.5, max_delay_s=30) == "ok"
    assert sleeps == [6.75]
    sleeps.clear(); calls.clear()
    r.retry_call(fn, operation="t", max_retries=2, base_delay_s=0.5, max_delay_s=3)
    assert sleeps == [3]
