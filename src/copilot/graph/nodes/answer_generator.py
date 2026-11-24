"""Node 7: build numbered evidence from ACCEPTED sources only and generate (or abstain)."""
from __future__ import annotations

from copilot.generation.answer import make_abstention
from copilot.generation.evidence import from_retrieved_chunks
from copilot.generation.models import EvidenceItem
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.config import Settings
from copilot.graph.state import CopilotState
from copilot.search.sanitizer import has_blocking_secret
from copilot.vectorstore.models import RetrievedChunk


_PUBLIC_ANGLE = {"external_current", "both"}


def partial_private_applies(state: CopilotState, settings: Settings) -> bool:
    """Experiment B (flag ``PARTIAL_PRIVATE_EVIDENCE``): may the grader's supporting chunks be used although it said insufficient?

    Only for a question the classifier calls ``both``, when the grader itself says public information is also needed, is confident
    at or above the private threshold, names valid supporting chunks and sees no contradiction. Never for an out-of-scope verdict.
    """
    if not settings.partial_private_evidence:
        return False
    grade = state.get("private_grade") or {}
    cls = state.get("classification") or {}
    return bool(
        cls.get("intent") == "both"
        and grade.get("needs_external_supplement")
        and not grade.get("contradiction_detected")
        and float(grade.get("confidence") or 0.0) >= float(grade.get("threshold") or settings.private_evidence_threshold)
        and grade.get("supporting_chunk_ids")
    )


def _search_note(state: CopilotState, settings: Settings) -> str | None:
    """One user-facing sentence on the web search, chosen from state flags (never from model or document text).

    Silent for questions with no public-information angle: a web search was never an option for them.
    """
    if state.get("external_triggered"):
        if str(state.get("external_blocked_reason") or "").startswith("search service unavailable"):
            return "The public web search service was unavailable."
        return "A public web search did not return a reliable, relevant source."
    cls = state.get("classification") or {}
    if cls.get("intent") not in _PUBLIC_ANGLE:
        return None
    if not settings.external_search_enabled:
        return "Public web search is switched off, so no public source was checked."
    if not state.get("external_search_allowed", True):
        return "Public web search was turned off for this request, so no public source was checked."
    if cls.get("contains_sensitive"):
        return "Public web search was not used because your message appears to contain sensitive information."
    if state.get("external_blocked_reason"):
        return "No public web search was run because no safe, generic public query could be formed from your question."
    return None


def _credential_note(state: CopilotState) -> str | None:
    """Tell the user to change a password or key they typed into the chat (pattern check on the question)."""
    if has_blocking_secret(state.get("question", "")):
        return "Your message seems to contain a password or secret; if it is real, please change it, because it was typed into this chat."
    return None


def _missing_text(state: CopilotState, settings: Settings) -> str:
    grade = state.get("private_grade") or {}
    parts = [m for m in grade.get("missing_information") or [] if m]
    text = "; ".join(parts) if parts else "the internal documents do not cover this question"
    notes = [n for n in (_search_note(state, settings), _credential_note(state)) if n]
    return text + (". " + " ".join(notes) if notes else "")


def make_answer_generator(deps: GraphDeps):
    """Build the ``answer_generator`` node."""
    model = deps.settings.groq_model

    @traced("answer_generator")
    def answer_generator(state: CopilotState) -> dict:
        forced = state.get("force_abstain")
        if state.get("fatal_error") and not forced:
            forced = {"reason": "internal_error", "error": state["fatal_error"]}
        out_of_scope = (state.get("classification") or {}).get("intent") == "out_of_scope"
        # An out-of-scope verdict stands unless private retrieval found evidence the grader accepted (finding 9, X18).
        if out_of_scope and not forced and not (state.get("private_grade") or {}).get("effective_sufficient"):
            forced = {"reason": "out_of_scope"}
        if forced:
            ans = make_abstention(forced["reason"], missing=forced.get("missing"), error=forced.get("error"), model=model)
            return {
                "answer": ans.model_dump(), "evidence": [], "needs_revision": False,
                "_event": event(f"abstaining: {forced['reason']}", "no LLM call; abstention message", reason=forced["reason"]),
            }

        grade = state.get("private_grade") or {}
        items: list[EvidenceItem] = []
        partial_private = False
        if grade.get("effective_sufficient"):
            by_id = {c["chunk_id"]: RetrievedChunk(**c) for c in state.get("retrieved_chunks", [])}
            selected = [by_id[i] for i in grade.get("selected_chunk_ids", []) if i in by_id]
            items = from_retrieved_chunks(selected)
        elif partial_private_applies(state, deps.settings):
            # EXPERIMENT B: the grader called the evidence insufficient only because part of the question needs current public
            # information; the chunks it found to support the other part still go to the generator (already access-filtered
            # at retrieval; no new retrieval path). The answer prompt already says to answer the supported part and state what is missing.
            by_id = {c["chunk_id"]: RetrievedChunk(**c) for c in state.get("retrieved_chunks", [])}
            selected = [by_id[i] for i in grade.get("supporting_chunk_ids", []) if i in by_id]
            items = from_retrieved_chunks(selected)
            partial_private = bool(items)
        for ext in state.get("accepted_external", []):
            items.append(EvidenceItem(
                ref=len(items) + 1, kind="external", label=ext["title"], location=ext["domain"], text=ext["content"],
                url=ext["url"], published_date=ext.get("published_date"),
            ))

        if not items:
            ans = make_abstention("insufficient_evidence", missing=_missing_text(state, deps.settings), model=model)
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
                uncited_lines=len(ans.uncited_lines), revision=revising, out_of_scope_verdict_overridden=out_of_scope,
                partial_private_used=partial_private,
            ),
        }

    return answer_generator
