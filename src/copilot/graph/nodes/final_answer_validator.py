"""Node 8: final grounding check. Revise at most MAX_VALIDATION_REVISIONS times, else abstain."""
from __future__ import annotations

from copilot.generation.answer import make_abstention
from copilot.generation.citations import EXTERNAL_WARNING, conflict_explained_by_status
from copilot.generation.evidence import format_evidence, neutralize_delimiters
from copilot.generation.models import EvidenceItem, GeneratedAnswer
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.graph.schemas import GroundingCheck
from copilot.graph.state import CopilotState
from copilot.llm.client import LLMUnavailableError, StructuredOutputError
from copilot.llm.prompts import VALIDATOR_SYSTEM_PROMPT


def external_separation_by_construction(ans: GeneratedAnswer) -> bool:
    """True when the answer cannot blend company guidance with public information (experiment A, a fact code can decide).

    It holds when every citation is external, each external citation carries a URL, and the rendered text starts with the
    verbatim external-information line. Code builds that line and the Sources block from real evidence, so an answer like this
    has no company-guidance claim for public information to be confused with. An answer that cites any private source is NOT
    covered: whether its two kinds of claim are kept apart is still left to the LLM check.
    """
    cites = ans.citations
    return bool(
        ans.used_external and cites
        and all(c.kind == "external" and c.url for c in cites)
        and ans.rendered.startswith(EXTERNAL_WARNING)
    )


def make_final_answer_validator(deps: GraphDeps):
    """Build the ``final_answer_validator`` node."""
    s = deps.settings

    @traced("final_answer_validator")
    def final_answer_validator(state: CopilotState) -> dict:
        raw = state.get("answer")
        ans = GeneratedAnswer(**raw) if raw else make_abstention("internal_error", model=s.groq_model)
        if ans.abstained:
            return {
                "answer": ans.model_dump(), "needs_revision": False,
                "validation": {"passed": True, "skipped": "abstention contains no claims"},
                "_event": event("abstention: nothing to validate", "pass"),
            }

        if not s.agent_use_validator:
            # ABLATION: no LLM grounding check; the answer goes out as generated (citations were still resolved in code).
            return {
                "answer": ans.model_dump(), "needs_revision": False,
                "validation": {"passed": True, "skipped": "ablation switch AGENT_USE_VALIDATOR=false: no grounding check was run"},
                "_event": event("validation skipped (ablation)", "pass without a grounding check", ablation="agent_use_validator=false"),
            }

        items = [EvidenceItem(**e) for e in state.get("evidence", [])]
        grade = state.get("private_grade") or {}
        conflict_in_evidence = bool(grade.get("effective_sufficient") and grade.get("contradiction_detected"))
        # A conflict between a current document and the older one it replaced is already explained to the reader by
        # the "Outdated source:" notice that code adds, so the model need not judge it (found by DEMO.md S5, where the
        # correct answer was rejected twice because the model wanted the word "conflict").
        explained_by_status = conflict_in_evidence and conflict_explained_by_status(ans.citations)
        conflict_expected = conflict_in_evidence and not explained_by_status
        superseded_present = any(i.status == "superseded" for i in items)
        problems: list[str] = []
        if ans.used_external and EXTERNAL_WARNING not in ans.rendered:
            problems.append("external information is not labelled with the required warning")

        prompt = (
            f"conflict_expected: {str(conflict_expected).lower()}\n"
            f"superseded_evidence_present: {str(superseded_present).lower()}\n\n"
            f"Evidence:\n{format_evidence(items, s.evidence_max_chars)}\n\n"
            f"DRAFT ANSWER:\n<answer>\n{neutralize_delimiters(ans.answer)}\n</answer>"
        )
        try:
            check, usage = deps.llm.structured(
                [{"role": "system", "content": VALIDATOR_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], GroundingCheck
            )
        except (LLMUnavailableError, StructuredOutputError) as exc:
            # Fail closed: an unverified answer is not delivered.
            refusal = make_abstention("validation_failed", error=str(exc), model=s.groq_model, usage=ans.usage)
            return {
                "answer": refusal.model_dump(), "needs_revision": False,
                "validation": {"passed": False, "error": str(exc), "rejected_answer": ans.rendered},
                "_event": event(f"validator unavailable: {type(exc).__name__}", "abstain (fail closed)", error=str(exc)),
            }

        if not check.grounded or check.unsupported_claims:
            problems.append("unsupported claims: " + "; ".join(check.unsupported_claims or ["unspecified"]))
        if conflict_expected and not check.conflicts_communicated:
            problems.append("the sources conflict but the answer does not clearly state each version with its citation")
        separation_in_code = s.external_label_check_in_code and external_separation_by_construction(ans)
        if ans.used_external and not check.external_clearly_separated and not separation_in_code:
            problems.append("public vendor information is not clearly separated from company guidance")
        if superseded_present and check.superseded_presented_as_current:
            problems.append("information from a superseded document is presented as the current rule; use the current document for what applies now and label the older version as superseded")

        details = {"grounded": check.grounded, "unsupported_claims": check.unsupported_claims,
                   "conflicts_communicated": check.conflicts_communicated, "conflict_expected": conflict_expected,
                   "conflict_explained_by_status": explained_by_status,
                   "external_clearly_separated": check.external_clearly_separated,
                   "external_separation_decided_in_code": bool(separation_in_code),
                   "superseded_presented_as_current": check.superseded_presented_as_current, "reasoning": check.reasoning,
                   "problems": problems}
        if not problems:
            return {
                "needs_revision": False, "validator_feedback": "", "llm_usage": usage_update(usage),
                "validation": {"passed": True, **details},
                "_event": event("answer is grounded", "pass", **details),
            }

        revisions = state.get("revision_count", 0)
        feedback = " | ".join(problems)
        if revisions < s.max_validation_revisions:
            return {
                "needs_revision": True, "revision_count": revisions + 1, "validator_feedback": feedback,
                "llm_usage": usage_update(usage), "validation": {"passed": False, "revising": True, **details},
                "_event": event(f"validation failed ({len(problems)} problem(s))", f"revise answer once ({revisions + 1}/{s.max_validation_revisions})", **details),
            }
        refusal = make_abstention("validation_failed", model=s.groq_model, usage=ans.usage)
        return {
            "answer": refusal.model_dump(), "needs_revision": False, "llm_usage": usage_update(usage),
            "validation": {"passed": False, "revising": False, "rejected_answer": ans.rendered, **details},
            "_event": event("validation failed after revision limit", "abstain", **details),
        }

    return final_answer_validator
