"""Node 1: classify the question (intent, sensitivity). Never replaces private-first retrieval."""
from __future__ import annotations

from copilot.generation.evidence import neutralize_delimiters
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.graph.schemas import QueryClassification
from copilot.graph.state import CopilotState
from copilot.llm.client import LLMUnavailableError, StructuredOutputError
from copilot.llm.prompts import CLASSIFIER_SYSTEM_PROMPT
from copilot.search.sanitizer import has_blocking_secret


def make_query_classifier(deps: GraphDeps):
    """Build the ``query_classifier`` node."""

    @traced("query_classifier")
    def query_classifier(state: CopilotState) -> dict:
        question = state["question"]
        usage = None
        fallback = False
        try:
            cls, usage = deps.llm.structured(
                [
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<question>\n{neutralize_delimiters(question)}\n</question>"},
                ],
                QueryClassification,
            )
        except (LLMUnavailableError, StructuredOutputError) as exc:
            # Fail safe: treat as an internal question and forbid external search.
            fallback = True
            cls = QueryClassification(
                intent="internal_it", contains_sensitive=True, confidence=0.0,
                sensitive_reason="classifier unavailable (fail-safe: external search blocked)",
                reasoning=f"classifier failed: {type(exc).__name__}",
            )
        if has_blocking_secret(question) and not cls.contains_sensitive:
            cls = cls.model_copy(update={
                "contains_sensitive": True, "sensitive_reason": "possible secret or credential detected by pattern check",
            })
        decision = "out_of_scope: skip retrieval and search" if cls.intent == "out_of_scope" else "continue: private retrieval first"
        return {
            "classification": cls.model_dump(),
            "current_query": question,
            "retry_count": 0,
            "llm_usage": usage_update(usage),
            "_event": event(
                f"intent={cls.intent}, sensitive={cls.contains_sensitive}, needs_current_public_info={cls.needs_current_public_info}",
                decision, intent=cls.intent, confidence=cls.confidence, sensitive=cls.contains_sensitive,
                sensitive_reason=cls.sensitive_reason, reasoning=cls.reasoning, fallback_used=fallback,
            ),
        }

    return query_classifier
