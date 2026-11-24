"""Experiment C, dev-only step: does expanding a question with paraphrases (merged by best score) rank the expected documents better?

Design and decision rules are in docs/POST_DAY3_EXPERIMENTS.md (Amendment 3), written BEFORE the first run. This module is the pure
part: ranking one arm for one question, and the rule that chooses between one and two paraphrases. Retrieval itself is the application's
own (``Retriever.retrieve`` per query, ``merge_retrievals`` for the merge), so a paraphrase can never reach a chunk the role may not read.

Arms: ``control`` (original question only, exactly ``rank_case``), ``para1`` (original + first paraphrase), ``para2`` (original + two).
Only DOCUMENT metrics decide; chunk relevance is keyword-defined and favours lexical matching, so it is reported and never used.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from copilot.evaluation.hybrid_ab import DOC_METRICS, paired_rows
from copilot.evaluation.results import RetrievedItem
from copilot.evaluation.retrieval_metrics import score_retrieval
from copilot.evaluation.schemas import EvalCase
from copilot.vectorstore.retriever import merge_retrievals

ARMS = ("control", "para1", "para2")
TIE_TOLERANCE = 0.005  # mean document MRR difference below which the cheaper arm (one paraphrase) is kept
OUTCOMES = ("chosen", "rejected_on_dev")


def paraphrases_for(arm: str, paraphrases: Sequence[str]) -> list[str]:
    """The paraphrases an arm uses (none for the control)."""
    return list(paraphrases[: {"control": 0, "para1": 1, "para2": 2}[arm]])


def rank_arm(case: EvalCase, retriever: Any, pool: int, paraphrases: Sequence[str]) -> list[RetrievedItem]:
    """Ranked pool for one question: the original query plus ``paraphrases``, merged by best cosine score (no score floor, no LLM)."""
    queries = [case.question, *paraphrases]
    merged = merge_retrievals([retriever.retrieve(q, case.role, top_k=pool, min_score=0.0) for q in queries], pool)
    return [RetrievedItem(document_name=c.document_name, chunk_id=c.chunk_id, score=c.score, text=c.text) for c in merged.chunks]


def run_arms(cases: Sequence[EvalCase], retriever: Any, ks: list[int], pool: int, paraphrases_by_case: Mapping[str, Sequence[str]]) -> dict[str, dict[str, dict[str, float | None]]]:
    """``{arm: {case_id: scores}}`` for every question that names expected documents."""
    out: dict[str, dict[str, dict[str, float | None]]] = {arm: {} for arm in ARMS}
    for case in cases:
        if not case.expected_documents:
            continue
        for arm in ARMS:
            items = rank_arm(case, retriever, pool, paraphrases_for(arm, paraphrases_by_case.get(case.id, ())))
            out[arm][case.id] = score_retrieval(case, items, ks)
    return out


def _mean(values: Sequence[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


def choose_arm(per_arm: Mapping[str, Mapping[str, Mapping[str, float | None]]], ids: Sequence[str], *, resamples: int = 2000, seed: int = 7, confidence: float = 0.95) -> dict[str, Any]:
    """The pre-registered choice rule on the dev questions.

    * An arm is ``harmful`` if any document metric versus the control is ``worse`` (interval excludes zero, at least 10 questions).
    * Among arms that are not harmful, take the one with the higher mean ``doc_mrr``; if the two differ by less than ``TIE_TOLERANCE``, take ``para1``.
    * If both arms are harmful, the outcome is ``rejected_on_dev``: experiment C stops and nothing is run on fresh-2.
    This only chooses between one and two paraphrases. It adopts nothing and proves no benefit.
    """
    arms: dict[str, Any] = {}
    for arm in ("para1", "para2"):
        rows = paired_rows(per_arm, ids, "control", arm, DOC_METRICS, resamples=resamples, seed=seed, confidence=confidence)
        worse = [r["metric"] for r in rows if r["verdict"] == "worse"]
        better = [r["metric"] for r in rows if r["verdict"] == "better"]
        arms[arm] = {"rows": rows, "worse": worse, "better": better, "harmful": bool(worse),
                     "mean_doc_mrr": _mean([per_arm[arm][i].get("doc_mrr") for i in ids if i in per_arm[arm]])}
    ok = [a for a in ("para1", "para2") if not arms[a]["harmful"]]
    if not ok:
        return {"outcome": "rejected_on_dev", "chosen": None, "arms": arms,
                "reason": "both arms made a document metric worse with an interval that excludes zero; experiment C stops here."}
    chosen = ok[0]
    if len(ok) == 2 and (arms["para2"]["mean_doc_mrr"] or 0.0) - (arms["para1"]["mean_doc_mrr"] or 0.0) >= TIE_TOLERANCE:
        chosen = "para2"
    return {"outcome": "chosen", "chosen": chosen, "arms": arms,
            "reason": f"{chosen} has no worse document metric and the higher (or tied, so cheaper) mean doc_mrr; this is a dev-only choice of the paraphrase count, not a benefit claim."}
