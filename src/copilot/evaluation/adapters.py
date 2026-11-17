"""Convert baseline / agentic responses into :class:`CaseResult` records (pure mapping, no I/O)."""
from __future__ import annotations

from copilot.evaluation.results import CaseResult
from copilot.evaluation.schemas import EvalCase
from copilot.generation.evidence import from_retrieved_chunks
from copilot.generation.models import EvidenceItem, GeneratedAnswer
from copilot.logging_config import redact
from copilot.validation.citations import MARKER_RE

# Abstentions caused by a failing service, not by the system deciding the evidence was insufficient.
# They are recorded as errors so an outage can never be scored as a "correct abstention".
INFRASTRUCTURE_REASONS = frozenset(
    {"llm_unavailable", "retrieval_unavailable", "grading_unavailable", "invalid_llm_output", "internal_error"}
)


def answer_text(ans: GeneratedAnswer) -> str:
    """The text that is scored: external warning + body without citation markers, but WITHOUT the Sources block.

    Source lines list document names, which could satisfy keyword checks by accident. Abstentions keep their
    full rendered text so a leak inside a refusal is still detected.
    """
    if ans.abstained or not ans.answer.strip():
        return ans.rendered
    head, sep, _ = ans.rendered.rpartition("\n\nSources:")
    return MARKER_RE.sub("", head if sep else ans.rendered).strip()


def _answer_fields(ans: GeneratedAnswer) -> dict:
    return {
        "answer": answer_text(ans), "abstained": ans.abstained, "abstention_reason": ans.abstention_reason,
        "used_external": ans.used_external,
        "cited_documents": sorted({c.label for c in ans.citations if c.kind == "private"}),
        "external_urls": [c.url for c in ans.citations if c.kind == "external" and c.url],
    }


def _error_for(ans: GeneratedAnswer, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if ans.abstained and ans.abstention_reason in INFRASTRUCTURE_REASONS:
        return redact(f"infrastructure: {ans.abstention_reason}" + (f" ({ans.error})" if ans.error else ""))[:300]
    return None


def from_baseline(case: EvalCase, resp) -> tuple[CaseResult, GeneratedAnswer, list[EvidenceItem]]:
    """Map a ``RAGResponse``. Returns the result, the answer and the evidence the judge should see."""
    ans = resp.answer
    result = CaseResult(
        case_id=case.id, category=case.category, mode="baseline", question=case.question, role=case.role,
        request_id=resp.request_id, path=["private_retriever", "answer_generator"], latency_ms=resp.latency_ms,
        prompt_tokens=ans.usage.prompt_tokens, completion_tokens=ans.usage.completion_tokens, llm_calls=ans.usage.calls,
        error=_error_for(ans, resp.error), **_answer_fields(ans),
    )
    evidence = from_retrieved_chunks(resp.retrieval.chunks) if resp.retrieval else []
    return result, ans, evidence


def from_agentic(case: EvalCase, resp) -> tuple[CaseResult, GeneratedAnswer, list[EvidenceItem]]:
    """Map an ``AgenticResponse``."""
    ans = resp.answer
    error = _error_for(ans, resp.error)
    grade = resp.private_grade or {}
    # The decision the graph acted on (LLM verdict AND threshold AND valid support), not the raw LLM flag.
    grade_sufficient = None if (error or not grade) else bool(grade.get("effective_sufficient", grade.get("sufficient")))
    validation = resp.validation or {}
    result = CaseResult(
        case_id=case.id, category=case.category, mode="agentic", question=case.question, role=case.role,
        request_id=resp.request_id, path=list(resp.path), external_triggered=resp.external_search_triggered,
        tavily_calls=resp.tavily_calls, retry_count=resp.retry_count, rewritten_queries=list(resp.rewritten_queries),
        grade_sufficient=grade_sufficient, grade_confidence=None if error or not grade else grade.get("confidence"),
        validation_passed=validation.get("passed") if validation else None, latency_ms=resp.latency_ms,
        prompt_tokens=resp.llm_usage.prompt_tokens, completion_tokens=resp.llm_usage.completion_tokens,
        llm_calls=resp.llm_usage.calls, error=error, **_answer_fields(ans),
    )
    return result, ans, [EvidenceItem(**e) for e in resp.evidence]


def error_result(case: EvalCase, mode: str, exc: BaseException, latency_ms: float) -> CaseResult:
    """A result for a request that raised (recorded as an error; the run continues)."""
    return CaseResult(
        case_id=case.id, category=case.category, mode=mode, question=case.question, role=case.role,  # type: ignore[arg-type]
        latency_ms=latency_ms, error=redact(f"{type(exc).__name__}: {exc}")[:300],
    )
