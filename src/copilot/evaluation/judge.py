"""LLM-as-judge for faithfulness and answer relevance (optional; deterministic checks stay primary).

The judge sees the question, the evidence the answer was generated from, and the answer. Its numbers
are a noisy signal from the same model family that wrote the answer, so treat them as supporting
evidence, not ground truth.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from copilot.config import Settings, get_settings
from copilot.evaluation.results import JudgeScores
from copilot.generation.evidence import format_evidence, neutralize_delimiters
from copilot.generation.models import EvidenceItem
from copilot.llm.client import LLMUnavailableError, LLMUsage, StructuredOutputError
from copilot.llm.prompts import JUDGE_SYSTEM_PROMPT
from copilot.logging_config import get_logger

log = get_logger(__name__)


class JudgeVerdict(BaseModel):
    """Structured judge output."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


class AnswerJudge:
    """Scores one answer. Judge failures return ``None`` scores; they never abort an evaluation run."""

    def __init__(self, llm, settings: Settings | None = None) -> None:
        self._llm = llm  # needs .structured(messages, schema) -> (parsed, LLMUsage)
        self._settings = settings or get_settings()

    def judge(self, question: str, answer_body: str, evidence: list[EvidenceItem]) -> tuple[JudgeScores | None, LLMUsage]:
        """Return ``(scores, usage)``; ``scores`` is None when the judge call failed."""
        user = (
            f"<question>\n{neutralize_delimiters(question)}\n</question>\n\n"
            f"Evidence:\n{format_evidence(evidence, self._settings.evidence_max_chars)}\n\n"
            f"<answer>\n{neutralize_delimiters(answer_body)}\n</answer>"
        )
        try:
            verdict, usage = self._llm.structured(
                [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}], JudgeVerdict
            )
        except (LLMUnavailableError, StructuredOutputError) as exc:
            log.warning("Judge failed: %s", type(exc).__name__)
            return None, LLMUsage()
        return (
            JudgeScores(
                faithfulness=verdict.faithfulness, relevance=verdict.relevance,
                unsupported_claims=verdict.unsupported_claims[:10], reason=verdict.reason[:500],
            ),
            usage,
        )
