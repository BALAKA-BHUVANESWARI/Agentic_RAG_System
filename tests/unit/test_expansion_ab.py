"""Experiment C dev step: arms and the pre-registered choice rule (pure, no network)."""
from __future__ import annotations

from copilot.evaluation import expansion_ab as ab
from copilot.evaluation.schemas import AnswerPoint, EvalCase
from copilot.vectorstore.models import RetrievalResult, RetrievedChunk


def case(cid: str, doc: str = "A.txt") -> EvalCase:
    return EvalCase(id=cid, category="direct_kb", question=f"question {cid}", expected_route="private_only", expected_documents=[doc],
                    expected_answer_points=[AnswerPoint(text="x", keywords=["x"])], expected_citations=[doc], expected_abstention=False)


def chunk(cid: str, doc: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=doc, document_name=doc, text="x", score=score)


class Stub:
    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    def retrieve(self, query, role=None, **kw):
        self.calls.append((query, kw.get("top_k"), kw.get("min_score")))
        ch = self.by_query.get(query, [])
        return RetrievalResult(query=query, role="employee", top_k=5, status="ok" if ch else "empty", chunks=ch)


def test_paraphrases_for_each_arm():
    assert ab.paraphrases_for("control", ["a", "b"]) == [] and ab.paraphrases_for("para1", ["a", "b"]) == ["a"]
    assert ab.paraphrases_for("para2", ["a"]) == ["a"] and ab.paraphrases_for("para2", ["a", "b"]) == ["a", "b"]


def test_rank_arm_merges_the_paraphrase_results_and_uses_no_score_floor():
    c = case("Q1")
    stub = Stub({"question Q1": [chunk("o:1", "B.txt", 0.4)], "para": [chunk("a:1", "A.txt", 0.7)]})
    items = ab.rank_arm(c, stub, 5, ["para"])
    assert [i.document_name for i in items] == ["A.txt", "B.txt"]
    assert all(call[1:] == (5, 0.0) for call in stub.calls)                       # pool as top_k, min_score 0.0, like rank_case


def test_run_arms_skips_questions_without_expected_documents_and_scores_every_arm():
    no_docs = EvalCase(id="N1", category="unanswerable", question="q", expected_route="abstain", expected_abstention=True)
    stub = Stub({"question Q1": [chunk("o:1", "B.txt", 0.4)], "p1": [chunk("a:1", "A.txt", 0.7)]})
    out = ab.run_arms([case("Q1"), no_docs], stub, [1, 3, 5], 5, {"Q1": ["p1", "p2"]})
    assert set(out) == set(ab.ARMS) and list(out["control"]) == ["Q1"]
    assert out["control"]["Q1"]["doc_mrr"] == 0.0 and out["para1"]["Q1"]["doc_mrr"] == 1.0 and out["para2"]["Q1"]["doc_mrr"] == 1.0


def _per_arm(control, para1, para2):
    ids = [f"Q{i}" for i in range(len(control))]
    mk = lambda vals: {i: {m: v for m in ab.DOC_METRICS} for i, v in zip(ids, vals)}  # noqa: E731
    return {"control": mk(control), "para1": mk(para1), "para2": mk(para2)}, ids


def test_choice_rejects_when_both_arms_are_clearly_worse():
    per, ids = _per_arm([1.0] * 12, [0.0] * 12, [0.0] * 12)
    out = ab.choose_arm(per, ids, resamples=200)
    assert out["outcome"] == "rejected_on_dev" and out["chosen"] is None


def test_choice_keeps_one_paraphrase_on_a_tie_and_two_when_clearly_higher():
    tie, ids = _per_arm([0.5] * 12, [0.5] * 12, [0.5] * 12)
    assert ab.choose_arm(tie, ids, resamples=200)["chosen"] == "para1"
    higher, ids = _per_arm([0.5] * 12, [0.5] * 12, [0.5] * 6 + [0.9] * 6)
    assert ab.choose_arm(higher, ids, resamples=200)["chosen"] == "para2"


def test_choice_skips_a_harmful_arm_and_picks_the_other():
    per, ids = _per_arm([1.0] * 12, [1.0] * 12, [0.2] * 12)
    out = ab.choose_arm(per, ids, resamples=200)
    assert out["outcome"] == "chosen" and out["chosen"] == "para1" and out["arms"]["para2"]["harmful"] is True
