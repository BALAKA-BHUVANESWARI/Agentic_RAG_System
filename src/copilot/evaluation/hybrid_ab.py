"""Hybrid-retrieval experiment: dense retrieval vs BM25 vs the two fused with Reciprocal Rank Fusion, on the labelled questions.

No LLM and no API tokens (it uses the existing Pinecone index and the local embedding model). The design, the fixed
parameters and the decision rule were written down BEFORE the first run in docs/HYBRID_RETRIEVAL.md; nothing here is tuned
on the questions.

* The dense arm is the application's own retrieval for the ORIGINAL question (``rank_case``, the same call the regression
  check and the evaluation runner use), so it can be compared with the committed retrieval baseline.
* The BM25 arm searches every chunk the role may read (the access filter is applied BEFORE ranking, like Pinecone's metadata
  filter) over the same text that is embedded (``title | section`` plus the chunk text).
* The hybrid arm fuses the two ranked lists with RRF. Unauthorised chunks cannot enter through either list.
* Only document-level metrics decide the outcome. Chunk-level relevance is defined by keywords from the expected answer points,
  which favours lexical matching, so chunk metrics are reported but flagged and never used for the decision.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from copilot.auth.access import allowed_access_levels
from copilot.config import Settings
from copilot.evaluation.paired import MIN_CASES_FOR_VERDICT, verdict_of
from copilot.evaluation.results import RetrievedItem
from copilot.evaluation.retrieval_metrics import rank_case, score_retrieval
from copilot.evaluation.schemas import EvalCase
from copilot.evaluation.stats import paired_bootstrap
from copilot.vectorstore.lexical import BM25Index, reciprocal_rank_fusion
from copilot.vectorstore.models import RetrievedChunk

SYSTEMS = ("dense", "bm25", "hybrid")
LABELS = {"dense": "Dense (the app's retriever)", "bm25": "BM25 only", "hybrid": "Hybrid: dense + BM25, RRF"}
# Fixed in advance. Changing any of these is a NEW experiment (tune on dev, judge on test; see docs/HYBRID_RETRIEVAL.md).
PARAMS: dict[str, float | int] = {"bm25_k1": 1.5, "bm25_b": 0.75, "rrf_k": 60}
DOC_METRICS = ("doc_recall@1", "doc_recall@3", "doc_recall@5", "doc_mrr", "doc_ndcg@5")  # decide the outcome
CHUNK_METRICS = ("chunk_mrr", "chunk_ndcg@5")  # reported only: relevance is keyword-defined, which favours lexical matching
MIN_BETTER_METRICS = 2  # five correlated document metrics are tested, so one 'better' among them is not enough
OUTCOMES = ("evidence_of_benefit", "no_evidence_of_benefit", "evidence_of_harm")


# ------------------------------------------------------------------------------ corpus
def lexical_text(chunk: RetrievedChunk) -> str:
    """The text that is embedded for a chunk (``title | section`` then the chunk text), so both arms see the same words."""
    prefix = chunk.title if not chunk.section else f"{chunk.title} | {chunk.section}"
    return f"{prefix}\n{chunk.text}"


class LexicalCorpus:
    """Every chunk of the store, plus a BM25 index over their embedded text."""

    def __init__(self, chunks: Iterable[RetrievedChunk], *, k1: float = float(PARAMS["bm25_k1"]), b: float = float(PARAMS["bm25_b"])) -> None:
        self.chunks: dict[str, RetrievedChunk] = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Index({cid: lexical_text(c) for cid, c in self.chunks.items()}, k1=k1, b=b)

    def describe(self) -> dict[str, Any]:
        """Provenance for the report: which documents and how many chunks the lexical index was built from."""
        per_doc: dict[str, int] = {}
        for c in self.chunks.values():
            per_doc[c.document_name] = per_doc.get(c.document_name, 0) + 1
        return {"chunks": len(self.chunks), "documents": dict(sorted(per_doc.items()))}


# ------------------------------------------------------------------------------ ranking
def _item(chunk: RetrievedChunk, score: float) -> RetrievedItem:
    return RetrievedItem(document_name=chunk.document_name, chunk_id=chunk.chunk_id, score=score, text=chunk.text)


def rank_systems(case: EvalCase, retriever: Any, corpus: LexicalCorpus, settings: Settings, *, pool: int, rrf_k: int = int(PARAMS["rrf_k"])) -> dict[str, list[RetrievedItem]]:
    """Ranked items per system for one question. All three only ever contain chunks the case's role may read."""
    dense = rank_case(retriever, case, pool, list(settings.eval_ks)).items
    levels = set(allowed_access_levels(case.role, settings))
    lexical = corpus.bm25.search(case.question, pool, allowed=lambda cid: corpus.chunks[cid].access_level in levels)
    bm25 = [_item(corpus.chunks[cid], score) for cid, score in lexical]
    by_id = {i.chunk_id: i for i in [*dense, *bm25]}
    fused = reciprocal_rank_fusion([[i.chunk_id for i in dense], [i.chunk_id for i in bm25]], k=rrf_k)[:pool]
    hybrid = [by_id[cid].model_copy(update={"score": score}) for cid, score in fused]
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid}


def run_experiment(cases: Sequence[EvalCase], retriever: Any, corpus: LexicalCorpus, settings: Settings, *, pool: int | None = None) -> dict[str, dict[str, dict[str, float | None]]]:
    """``{system: {case_id: scores}}`` for every question that names expected documents."""
    pool = pool or settings.eval_retrieval_pool
    out: dict[str, dict[str, dict[str, float | None]]] = {s: {} for s in SYSTEMS}
    for case in cases:
        if not case.expected_documents:
            continue
        for system, items in rank_systems(case, retriever, corpus, settings, pool=pool).items():
            out[system][case.id] = score_retrieval(case, items, list(settings.eval_ks))
    return out


# ------------------------------------------------------------------------------ summary and decision
def _mean(values: Sequence[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


def paired_rows(per_system: Mapping[str, Mapping[str, Mapping[str, float | None]]], ids: Sequence[str], before: str, after: str, metrics: Sequence[str], *, resamples: int, seed: int, confidence: float) -> list[dict[str, Any]]:
    """Per-metric paired difference ``after - before`` over the questions in ``ids``, with a bootstrap interval and a verdict."""
    rows = []
    for metric in metrics:
        pairs = [(per_system[before][i][metric], per_system[after][i][metric]) for i in ids if i in per_system[before] and i in per_system[after]]
        pairs = [(b, a) for b, a in pairs if b is not None and a is not None]
        if not pairs:
            continue
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        stat = paired_bootstrap(xs, ys, resamples=resamples, seed=seed, confidence=confidence)
        rows.append({
            "metric": metric, "n": stat["n"], "before": sum(xs) / len(xs), "after": sum(ys) / len(ys), "diff": stat["diff"], "lo": stat["lo"], "hi": stat["hi"],
            "improved": sum(1 for b, a in pairs if a > b + 1e-9), "worsened": sum(1 for b, a in pairs if a < b - 1e-9), "verdict": verdict_of(stat, True),
        })
    return rows


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered rule, over the document-level rows of hybrid vs dense on all questions.

    * ``evidence_of_harm``: any document metric is ``worse`` (interval excludes zero, at least 10 questions);
    * ``evidence_of_benefit``: no metric is worse and at least ``MIN_BETTER_METRICS`` are ``better``;
    * otherwise ``no_evidence_of_benefit``, which is NOT evidence of no effect: the questions may simply be too few or dense retrieval
      already near the ceiling.

    Even ``evidence_of_benefit`` only licenses a follow-up (a production design and variants tuned on dev, judged on test). It does not adopt anything.
    """
    doc = [r for r in rows if r["metric"] in DOC_METRICS]
    better, worse = [r["metric"] for r in doc if r["verdict"] == "better"], [r["metric"] for r in doc if r["verdict"] == "worse"]
    if worse:
        outcome, reason = "evidence_of_harm", f"{', '.join(worse)} got worse with an interval that excludes zero."
    elif len(better) >= MIN_BETTER_METRICS:
        outcome, reason = "evidence_of_benefit", f"{', '.join(better)} got better with intervals that exclude zero, and none got worse."
    else:
        n = max((r["n"] for r in doc), default=0)
        outcome = "no_evidence_of_benefit"
        reason = (f"Fewer than {MIN_BETTER_METRICS} document metrics are distinguishable from zero improvement"
                  + (f" (only {n} questions: too few for a verdict)." if n < MIN_CASES_FOR_VERDICT else ".") + " This is not evidence that the arms are equal.")
    return {"outcome": outcome, "better": better, "worse": worse, "reason": reason, "rule": f"harm if any document metric is worse; benefit if at least {MIN_BETTER_METRICS} are better and none worse; document metrics: {', '.join(DOC_METRICS)}"}


def summarize(
    per_system: Mapping[str, Mapping[str, Mapping[str, float | None]]], cases: Sequence[EvalCase], subsets: Mapping[str, set[str]], *,
    resamples: int = 2000, seed: int = 7, confidence: float = 0.95,
) -> dict[str, Any]:
    """The experiment's report: means per system, paired differences against dense for each subset, changed questions, the decision."""
    scored = sorted(per_system["dense"])
    metrics = [*DOC_METRICS, *CHUNK_METRICS]
    by_case = {c.id: c for c in cases}
    out_subsets: dict[str, Any] = {}
    for name, members in {"all": set(scored), **subsets}.items():
        ids = [i for i in scored if i in members]
        out_subsets[name] = {
            "n": len(ids),
            "means": {s: {m: _mean([per_system[s][i].get(m) for i in ids]) for m in metrics} for s in SYSTEMS},
            "paired": {after: paired_rows(per_system, ids, "dense", after, metrics, resamples=resamples, seed=seed, confidence=confidence) for after in ("bm25", "hybrid")},
        }
    changed = {
        "improved": [i for i in scored if (per_system["hybrid"][i].get("doc_mrr") or 0) > (per_system["dense"][i].get("doc_mrr") or 0) + 1e-9],
        "worsened": [i for i in scored if (per_system["hybrid"][i].get("doc_mrr") or 0) < (per_system["dense"][i].get("doc_mrr") or 0) - 1e-9],
    }
    by_category: dict[str, Any] = {}
    for cat in sorted({by_case[i].category for i in scored}):
        ids = [i for i in scored if by_case[i].category == cat]
        by_category[cat] = {"n": len(ids), **{s: _mean([per_system[s][i].get("doc_mrr") for i in ids]) for s in SYSTEMS}}
    return {
        "params": dict(PARAMS), "confidence": confidence, "resamples": resamples, "seed": seed, "n_questions": len(scored),
        "subsets": out_subsets, "changed_vs_dense_by_doc_mrr": changed, "doc_mrr_by_category": by_category,
        "decision": decide(out_subsets["all"]["paired"]["hybrid"]),
        "caveats": [
            "Intervals describe which questions were asked, not repeat-run noise; the arms are deterministic.",
            "Chunk metrics use keyword-defined relevance, which favours lexical matching; they never decide the outcome.",
            "The corpus is 11 short synthetic documents; BM25 statistics on so small a corpus are noisy and the result may not transfer to a larger one.",
        ],
    }
