"""Node 6: grade public results (credibility, relevance, freshness, support) before any use."""
from __future__ import annotations

from datetime import datetime, timezone

from copilot.generation.evidence import format_evidence, neutralize_delimiters
from copilot.generation.models import EvidenceItem
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.graph.schemas import ExternalGrade
from copilot.graph.state import CopilotState
from copilot.llm.client import LLMUnavailableError, StructuredOutputError
from copilot.llm.prompts import EXTERNAL_GRADER_SYSTEM_PROMPT
from copilot.search.tavily_client import ExternalResult
from copilot.search.urls import is_user_generated_path

_NON_OFFICIAL_MIN_CREDIBILITY = 0.8


def _items(results: list[ExternalResult]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            ref=i, kind="external", label=r.title, location=r.domain, text=r.content, url=r.url,
            published_date=r.published_date,
        )
        for i, r in enumerate(results, start=1)
    ]


def make_external_evidence_grader(deps: GraphDeps):
    """Build the ``external_evidence_grader`` node."""
    s = deps.settings

    @traced("external_evidence_grader")
    def external_evidence_grader(state: CopilotState) -> dict:
        results = [ExternalResult(**r) for r in state.get("external_results", [])]
        if not results:
            return {
                "external_grade": {"sufficient": False, "accepted_refs": [], "note": "no external results to grade"},
                "accepted_external": [],
                "_event": event("no external results to grade", "continue without external evidence"),
            }
        items = _items(results)
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            grade, usage = deps.llm.structured(
                [
                    {"role": "system", "content": EXTERNAL_GRADER_SYSTEM_PROMPT.format(today=today)},
                    {"role": "user", "content": (
                        f"<question>\n{neutralize_delimiters(state['question'])}\n</question>\n\n"
                        f"Search results:\n{format_evidence(items, s.external_content_max_chars)}"
                    )},
                ],
                ExternalGrade,
            )
        except (LLMUnavailableError, StructuredOutputError) as exc:
            return {
                "external_grade": {"sufficient": False, "accepted_refs": [], "error": str(exc)},
                "accepted_external": [],
                "_event": event(f"external grader failed: {type(exc).__name__}", "continue without external evidence", error=str(exc)),
            }

        by_ref = {g.ref: g for g in grade.results}
        accepted, details = [], []
        for ref, r in enumerate(results, start=1):
            g = by_ref.get(ref)
            # A community Q&A thread is written by users, whatever credibility the model gives it
            # (it once scored a forum post 1.0). Decided by URL in code, not by the model.
            community = is_user_generated_path(r.url, s.user_generated_url_markers)
            ok = bool(
                not community
                and g and g.accept and g.supports_answer and g.freshness != "stale"
                and g.relevance >= s.external_evidence_threshold
                and (r.preferred or g.credibility >= _NON_OFFICIAL_MIN_CREDIBILITY)
            )
            reason = g.reason if g else "not graded"
            if community:
                reason = f"user-written community page, never accepted as evidence (grader said: {reason})"
            details.append({"ref": ref, "url": r.url, "official": r.preferred, "community_page": community, "accepted": ok,
                            "relevance": g.relevance if g else None, "credibility": g.credibility if g else None,
                            "freshness": g.freshness if g else None, "reason": reason})
            if ok:
                accepted.append({**r.model_dump(), "grade": g.model_dump()})
        effective = bool(accepted) and grade.confidence >= s.external_evidence_threshold
        if not effective:
            accepted = []
        return {
            "external_grade": {**grade.model_dump(), "accepted_refs": [d["ref"] for d in details if d["accepted"]] if effective else [],
                               "effective_sufficient": effective, "details": details},
            "accepted_external": accepted,
            "llm_usage": usage_update(usage),
            "_event": event(
                f"{len(accepted)}/{len(results)} external result(s) accepted (confidence {grade.confidence:.2f})",
                "use external evidence (labelled)" if accepted else "reject external evidence",
                details=details, contradiction_detected=grade.contradiction_detected, reasoning=grade.reasoning,
            ),
        }

    return external_evidence_grader
