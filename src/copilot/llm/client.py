"""Groq LLM client: plain chat and Pydantic-validated structured output.

* Bounded retries with backoff for transient errors (connection, rate limit, 5xx).
* Any final failure is raised as :class:`LLMUnavailableError` (messages redacted) so
  callers can degrade gracefully instead of crashing.
* Structured calls use JSON mode, validate with Pydantic, and repair-retry ONCE;
  after that :class:`StructuredOutputError` is raised and the caller falls back to a
  safe default.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from copilot.config import Settings, get_settings
from copilot.llm.prompts import json_schema_instruction
from copilot.logging_config import get_logger, redact
from copilot.retry import retry_call

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)
Messages = Sequence[dict[str, str]]


class LLMUnavailableError(RuntimeError):
    """The LLM service failed after bounded retries (or rejected the request)."""


class StructuredOutputError(ValueError):
    """The model did not return valid JSON for the requested schema (after one repair)."""


class LLMUsage(BaseModel):
    """Token usage and call count (aggregated across calls where needed)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "LLMUsage") -> None:
        """Accumulate another usage record into this one."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls


class LLMResponse(BaseModel):
    """One chat completion."""

    text: str
    usage: LLMUsage
    model: str
    latency_ms: float


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object in ``text`` (tolerates code fences)."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


class GroqClient:
    """Thin, testable wrapper around the Groq SDK."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        retry_exceptions: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._retry_exceptions = retry_exceptions
        self.usage_total = LLMUsage()  # cumulative, useful for evaluation cost tracking

    def _get_client(self) -> Any:
        if self._client is None:
            from groq import Groq

            self._client = Groq(
                api_key=self._settings.require_secret("groq"),
                timeout=self._settings.groq_timeout_s,
                max_retries=0,  # retries are handled by copilot.retry for consistent logging
            )
        return self._client

    def _retryable(self) -> tuple[type[BaseException], ...]:
        if self._retry_exceptions is not None:
            return self._retry_exceptions
        import groq

        return (groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError)

    def list_models(self) -> list[str]:
        """IDs of models available to this API key."""
        return sorted(m.id for m in self._get_client().models.list().data)

    def chat(
        self,
        messages: Messages,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Run one chat completion with bounded retries."""
        s = self._settings
        client = self._get_client()  # config errors (missing key) propagate as RuntimeError
        kwargs: dict[str, Any] = {
            "model": s.groq_model,
            "messages": list(messages),
            "temperature": s.groq_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or s.llm_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if s.groq_reasoning_effort:  # sent via extra_body so it works across SDK versions
            kwargs["extra_body"] = {"reasoning_effort": s.groq_reasoning_effort}
        started = time.perf_counter()
        try:
            raw = retry_call(
                lambda: client.chat.completions.create(**kwargs),
                operation="groq.chat",
                max_retries=s.groq_max_retries,
                base_delay_s=s.service_retry_base_delay_s,
                retry_on=self._retryable(),
                max_delay_s=s.service_retry_max_delay_s,
            )
        except Exception as exc:  # noqa: BLE001 - every SDK/HTTP failure becomes one clear error
            raise LLMUnavailableError(f"{type(exc).__name__}: {redact(str(exc))[:200]}") from exc

        raw_usage = getattr(raw, "usage", None)
        usage = LLMUsage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
            calls=1,
        )
        self.usage_total.add(usage)
        text = (raw.choices[0].message.content or "") if getattr(raw, "choices", None) else ""
        return LLMResponse(
            text=text, usage=usage, model=getattr(raw, "model", s.groq_model) or s.groq_model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def structured(
        self, messages: Messages, schema: type[T], *, max_tokens: int | None = None
    ) -> tuple[T, LLMUsage]:
        """Get a Pydantic-validated object from the model (one repair retry).

        Returns ``(parsed, usage)`` where usage aggregates every call made.
        Raises :class:`StructuredOutputError` if both attempts are invalid and
        :class:`LLMUnavailableError` if the service fails.
        """
        msgs: list[dict[str, str]] = [{"role": "system", "content": json_schema_instruction(schema)}, *messages]
        total = LLMUsage()
        last_error = "unknown"
        for _attempt in range(2):
            response = self.chat(msgs, json_mode=True, max_tokens=max_tokens)
            total.add(response.usage)
            try:
                return schema.model_validate(parse_json_object(response.text)), total
            except ValueError as exc:  # JSONDecodeError and pydantic ValidationError both subclass ValueError
                last_error = type(exc).__name__
                log.warning("Structured output invalid (%s); %s", last_error, "retrying" if _attempt == 0 else "giving up")
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": response.text[:2000]},
                    {
                        "role": "user",
                        "content": "That reply was not valid JSON matching the required schema. "
                        "Reply again with ONLY the JSON object.",
                    },
                ]
        raise StructuredOutputError(f"Model did not return valid {schema.__name__} JSON ({last_error}).")
