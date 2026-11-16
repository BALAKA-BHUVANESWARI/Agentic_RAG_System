"""Node 7: build numbered evidence from ACCEPTED sources only and generate (or abstain)."""
from __future__ import annotations

from copilot.generation.answer import make_abstention
from copilot.generation.evidence import from_retrieved_chunks
from copilot.generation.models import EvidenceItem
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.graph.state import CopilotState
from copilot.vectorstore.models import RetrievedChunk


def _missing_text(state: CopilotState) -> str:
    grade = state.get("private_grade") or {}
    parts = [m for m in grade.get("missing_information") or [] if m]
    text = "; ".join(parts) if parts else "the internal documents do not cover this question"
    if state.get("external_triggered"):
        text += ". A public web search did not return a reliable, relevant source."
    return text


def make_answer_generator(deps: GraphDeps):
    """Build the ``answer_generator`` node."""
    model = deps.settings.groq_model

    @traced("answer_generator")
    def answer_generator(state: CopilotState) -> dict:
        forced = state.get("force_abstain")
        if state.get("fatal_error") and not forced:
            forced = {"reason": "internal_error", "error": state["fatal_error"]}
        if (state.get("classification") or {}).get("intent") == "out_of_scope" and not forced:
            forced = {"reason": "out_of_scope"}
        if forced:
            ans = make_abstention(forced["reason"], missing=forced.get("missing"), error=forced.get("error"), model=model)
            return {
                "answer": ans.model_dump(), "evidence": [], "needs_revision": False,
                "_event": event(f"abstaining: {forced['reason']}", "no LLM call; abstention message", reason=forced["reason"]),
            }

        grade = state.get("private_grade") or {}
        items: list[EvidenceItem] = []
        if grade.get("effective_sufficient"):
            by_id = {c["chunk_id"]: RetrievedChunk(**c) for c in state.get("retrieved_chunks", [])}
            selected = [by_id[i] for i in grade.get("selected_chunk_ids", []) if i in by_id]
            items = from_retrieved_chunks(selected)
        for ext in state.get("accepted_external", []):
            items.append(EvidenceItem(
                ref=len(items) + 1, kind="external", label=ext["title"], location=ext["domain"], text=ext["content"],
                url=ext["url"], published_date=ext.get("published_date"),
            ))

        if not items:
            ans = make_abstention("insufficient_evidence", missing=_missing_text(state), model=model)
            return {
                "answer": ans.model_dump(), "evidence": [], "needs_revision": False,
                "_event": event("no accepted evidence: abstaining", "abstain; recommend IT service desk",
                                private_effective=bool(grade.get("effective_sufficient")),
                                external_triggered=bool(state.get("external_triggered"))),
            }

        ans = deps.generator.generate(state["question"], items, feedback=state.get("validator_feedback"))
        revising = bool(state.get("validator_feedback"))
        return {
            "answer": ans.model_dump(),
            "evidence": [i.model_dump() for i in items],
            "needs_revision": False,
            "llm_usage": usage_update(ans.usage),
            "_event": event(
                f"{'revised ' if revising else ''}answer generated from {len(items)} evidence block(s)"
                + (" (abstained)" if ans.abstained else ""),
                f"abstain: {ans.abstention_reason}" if ans.abstained else "continue: validate grounding",
                private_sources=sum(i.kind == "private" for i in items), external_sources=sum(i.kind == "external" for i in items),
                citations=[c.ref for c in ans.citations], invalid_markers_removed=ans.invalid_markers_removed,
                uncited_lines=len(ans.uncited_lines), revision=revising,
            ),
        }

    return answer_generator
