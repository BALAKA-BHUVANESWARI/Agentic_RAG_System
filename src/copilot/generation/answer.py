"""Answer generation: grounded, cited answers from numbered evidence (or a safe abstention).

Reused by the baseline RAG and by the LangGraph ``answer_generator`` node.
"""
from __future__ import annotations

from pydantic import BaseModel

from copilot.config import Settings, get_settings
from copilot.generation.citations import build_citations, render_answer
from copilot.generation.evidence import format_evidence, neutralize_delimiters
from copilot.generation.models import AbstentionReason, EvidenceItem, GeneratedAnswer
from copilot.llm.client import GroqClient, LLMUnavailableError, LLMUsage, StructuredOutputError
from copilot.llm.prompts import ANSWER_SYSTEM_PROMPT
from copilot.logging_config import get_logger
from copilot.validation.citations import extract_markers, strip_invalid_markers, uncited_lines

log = get_logger(__name__)

_SERVICE_DESK = "Please contact the IT Service Desk for help."
_ABSTENTION_TEXT: dict[str, str] = {
    "no_evidence": f"I couldn't find reliable information in the available documents to answer this. {_SERVICE_DESK}",
    "insufficient_evidence": f"The available information does not establish an answer to this question. {_SERVICE_DESK}",
    "uncited_answer": f"I couldn't produce an answer that is backed by the available documents. {_SERVICE_DESK}",
    "invalid_llm_output": f"I couldn't produce a reliable answer right now. {_SERVICE_DESK}",
    "llm_unavailable": f"The answer service is temporarily unavailable, so I can't answer right now. Please try again shortly. {_SERVICE_DESK}",
    "retrieval_unavailable": f"The knowledge base is temporarily unavailable, so I can't answer right now. Please try again shortly. {_SERVICE_DESK}",
    "out_of_scope": "This assistant only answers IT support questions. For anything else, please use the appropriate company channel.",
    "grading_unavailable": f"I couldn't check the available documents reliably right now, so I can't answer safely. Please try again shortly. {_SERVICE_DESK}",
    "validation_failed": f"I couldn't verify that a draft answer was fully supported by the available sources, so I'm not providing it. {_SERVICE_DESK}",
    "internal_error": f"Something went wrong while processing your question. {_SERVICE_DESK}",
}


class AnswerDraft(BaseModel):
    """Structured output requested from the LLM."""

    can_answer: bool
    answer: str = ""
    missing_information: str | None = None


def make_abstention(
    reason: AbstentionReason,
    *,
    missing: str | None = None,
    usage: LLMUsage | None = None,
    model: str = "",
    error: str | None = None,
    invalid_markers_removed: list[int] | None = None,
) -> GeneratedAnswer:
    """Build an abstention answer (no claims, recommends the service desk)."""
    text = _ABSTENTION_TEXT[reason]
    if missing:
        text += f"\n\nWhat is missing: {missing.strip()}"
    return GeneratedAnswer(
        rendered=text, abstained=True, abstention_reason=reason, missing_information=missing,
        error=error, usage=usage or LLMUsage(), model=model,
        invalid_markers_removed=invalid_markers_removed or [],
    )


class AnswerGenerator:
    """Generates cited answers using Groq. Citations are resolved by code, not by the LLM."""

    def __init__(self, settings: Settings | None = None, llm: GroqClient | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or GroqClient(self._settings)

    def generate(self, question: str, evidence: list[EvidenceItem], feedback: str | None = None, *, _citation_retry: bool = False) -> GeneratedAnswer:
        """Answer ``question`` from ``evidence`` or abstain. Never raises for service failures.

        ``feedback`` (from the final validator) asks the model to revise a previous draft.
        A draft with no valid ``[n]`` marker is retried once with an explicit reminder; a second
        uncited draft abstains (``uncited_answer``). ``_citation_retry`` is internal.
        """
        model = self._settings.groq_model
        if not evidence:
            return make_abstention("no_evidence", model=model)

        user_prompt = (
            f"<question>\n{neutralize_delimiters(question.strip())}\n</question>\n\n"
            f"Evidence:\n{format_evidence(evidence, self._settings.evidence_max_chars)}"
        )
        if feedback:
            user_prompt += (
                "\n\nReviewer feedback on your previous draft (fix these problems; remove or correct "
                f"every unsupported claim, and use only the evidence above):\n{neutralize_delimiters(feedback)[:1500]}"
            )
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            draft, usage = self._llm.structured(messages, AnswerDraft)
        except LLMUnavailableError as exc:
            return make_abstention("llm_unavailable", model=model, error=str(exc))
        except StructuredOutputError as exc:
            return make_abstention("invalid_llm_output", model=model, error=str(exc))

        if not draft.can_answer or not draft.answer.strip():
            return make_abstention("insufficient_evidence", missing=draft.missing_information, usage=usage, model=model)

        valid = {e.ref for e in evidence}
        body, removed = strip_invalid_markers(draft.answer.strip(), valid)
        used = extract_markers(body)
        if not used:
            if not _citation_retry:
                log.warning("Answer had no valid citations; retrying once", extra={"removed": removed})
                reminder = (
                    "Your previous draft had no valid citation markers. Rewrite it so that every material claim "
                    "is followed by the evidence id in square brackets, such as [1], using only ids that exist above."
                )
                retry = self.generate(question, evidence, feedback=f"{feedback}\n{reminder}" if feedback else reminder, _citation_retry=True)
                total = usage.model_copy()
                total.add(retry.usage)
                return retry.model_copy(update={"usage": total})
            log.warning("Answer had no valid citations; abstaining", extra={"removed": removed})
            return make_abstention("uncited_answer", usage=usage, model=model, invalid_markers_removed=removed)

        citations = build_citations(evidence, used)
        return GeneratedAnswer(
            answer=body,
            rendered=render_answer(body, citations),
            citations=citations,
            used_external=any(c.kind == "external" for c in citations),
            invalid_markers_removed=removed,
            uncited_lines=uncited_lines(body, valid),
            usage=usage,
            model=model,
        )
