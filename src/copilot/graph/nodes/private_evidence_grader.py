"""Node 3: LLM grading of retrieved internal evidence, combined with configurable thresholds."""
from __future__ import annotations

from copilot.generation.evidence import format_evidence, from_retrieved_chunks, neutralize_delimiters
from copilot.graph.deps import GraphDeps
from copilot.graph.nodes.common import event, traced, usage_update
from copilot.graph.schemas import PrivateGrade
from copilot.graph.state import CopilotState
from copilot.llm.client import LLMUnavailableError, StructuredOutputError
from copilot.llm.prompts import PRIVATE_GRADER_SYSTEM_PROMPT
from copilot.vectorstore.models import RetrievedChunk


def make_private_evidence_grader(deps: GraphDeps):
    """Build the ``private_evidence_grader`` node."""
    s = deps.settings

    @traced("private_evidence_grader")
    def private_evidence_grader(state: CopilotState) -> dict:
        chunks = [RetrievedChunk(**c) for c in state.get("retrieved_chunks", [])]
        if not chunks:
            grade = PrivateGrade(
                sufficient=False, confidence=0.0, missing_information=["No relevant internal documents were retrieved."],
                reasoning="Nothing was retrieved, so the LLM grader was not called.", recommended_action="rewrite_query",
            )
            return _result(grade, [], [], s.private_evidence_threshold, None, "no chunks retrieved")

        items = from_retrieved_chunks(chunks)
        if not s.agent_use_grader:
            # ABLATION: no LLM call; every retrieved chunk is treated as supporting evidence (baseline-style).
            skipped = PrivateGrade(
                sufficient=True, confidence=1.0, supporting_refs=list(range(1, len(items) + 1)), recommended_action="answer",
                reasoning="Evidence grader skipped (ablation switch AGENT_USE_GRADER=false): all retrieved chunks are used.",
            )
            out = _result(skipped, items, list(range(1, len(items) + 1)), s.private_evidence_threshold, None, None)
            out["_event"]["summary"] = "grader skipped (ablation): all retrieved chunks used"
            out["_event"]["data"]["ablation"] = "agent_use_grader=false"
            out["private_grade"]["skipped"] = True
            return out
        try:
            grade, usage = deps.llm.structured(
                [
                    {"role": "system", "content": PRIVATE_GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"<question>\n{neutralize_delimiters(state['question'])}\n</question>\n\n"
                        f"Evidence:\n{format_evidence(items, s.evidence_max_chars)}"
                    )},
                ],
                PrivateGrade,
            )
        except (LLMUnavailableError, StructuredOutputError) as exc:
            failed = PrivateGrade(sufficient=False, confidence=0.0, reasoning=f"grader failed: {type(exc).__name__}", recommended_action="abstain")
            out = _result(failed, items, [], s.private_evidence_threshold, None, "grader unavailable")
            out["force_abstain"] = {"reason": "grading_unavailable", "error": str(exc)}
            out["_event"]["decision"] = "abstain: evidence could not be graded"
            return out

        valid_refs = [r for r in dict.fromkeys(grade.supporting_refs) if 1 <= r <= len(items)]
        out = _result(grade, items, valid_refs, s.private_evidence_threshold, usage, None)
        return out

    return private_evidence_grader


def _result(grade: PrivateGrade, items, valid_refs: list[int], threshold: float, usage, note: str | None) -> dict:
    effective = bool(grade.sufficient and grade.confidence >= threshold and valid_refs)
    selected = [items[r - 1].chunk_id for r in valid_refs]
    if grade.contradiction_detected:  # conflicting evidence must all reach the generator
        selected = [i.chunk_id for i in items]
    stored = {
        **grade.model_dump(),
        "supporting_chunk_ids": [items[r - 1].chunk_id for r in valid_refs],
        "selected_chunk_ids": selected if effective else [],
        "effective_sufficient": effective,
        "threshold": threshold,
    }
    if effective:
        decision = "sufficient: answer from internal documents"
    else:
        reasons = []
        if not grade.sufficient:
            reasons.append("grader says insufficient")
        elif grade.confidence < threshold:
            reasons.append(f"confidence {grade.confidence:.2f} below threshold {threshold:.2f}")
        elif not valid_refs:
            reasons.append("no valid supporting chunks")
        decision = "insufficient: " + (note or "; ".join(reasons))
    return {
        "private_grade": stored,
        "llm_usage": usage_update(usage),
        "_event": event(
            f"sufficient={grade.sufficient}, confidence={grade.confidence:.2f}, contradiction={grade.contradiction_detected}, effective={effective}",
            decision, recommended_action=grade.recommended_action, missing_information=grade.missing_information,
            contradiction_details=grade.contradiction_details, needs_external_supplement=grade.needs_external_supplement,
            supporting_chunks=stored["supporting_chunk_ids"], reasoning=grade.reasoning, threshold=threshold,
        ),
    }
