"""Lexical retrieval pieces for the hybrid-retrieval experiment: a small BM25 index and Reciprocal Rank Fusion.

Pure Python, standard library only. This module is NOT wired into the application's retriever: it exists so the
experiment in ``copilot.evaluation.hybrid_ab`` (docs/HYBRID_RETRIEVAL.md) can measure whether a lexical signal adds
anything to dense retrieval before any production change is considered.

* Tokenising: lower-case alphanumeric tokens, a short English stop-word list, and the classic "S-stemmer" (plural folding
  only, so ``passwords`` matches ``password`` without the surprises of a full stemmer).
* BM25 uses the Lucene form of the IDF (never negative): ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
* RRF (Cormack et al., 2009) fuses rankings without having to calibrate their scores against each other.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Mapping, Sequence

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in into is it its me my no not of on or our so that the their them then there "
    "these they this to was we were what when where which who why will with you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Harman's S-stemmer: fold simple plurals (``policies`` -> ``policy``, ``passwords`` -> ``password``), nothing else."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stop words, plurals folded. Digits are kept (``2fa``, ``ipv4``, ``8``)."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over a fixed corpus of ``{id: text}``."""

    def __init__(self, documents: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be >= 0 and b must be between 0 and 1")
        self.k1, self.b = k1, b
        self._tf: dict[str, Counter[str]] = {doc_id: Counter(tokenize(text)) for doc_id, text in documents.items()}
        self._len = {doc_id: sum(tf.values()) for doc_id, tf in self._tf.items()}
        self._avg = (sum(self._len.values()) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(self._tf)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def __len__(self) -> int:
        return len(self._tf)

    def score(self, query: str, doc_id: str) -> float:
        """BM25 score of one document for the query (0.0 when no query term occurs in it)."""
        tf, dl = self._tf[doc_id], self._len[doc_id]
        norm = self.k1 * (1 - self.b + self.b * dl / self._avg) if self._avg else self.k1
        total = 0.0
        for term in set(tokenize(query)):
            f = tf.get(term, 0)
            if f:
                total += self._idf[term] * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, query: str, top_k: int, allowed: Callable[[str], bool] | None = None) -> list[tuple[str, float]]:
        """Up to ``top_k`` ``(id, score)`` pairs, best first, ties broken by id. Documents with no query term are not returned.

        ``allowed(id)`` filters candidates BEFORE ranking; it is how a caller applies access control.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self._tf if allowed is None or allowed(doc_id)]
        scored = [(d, s) for d, s in scored if s > 0.0]
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60, weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: ``score(id) = sum_i w_i / (k + rank_i(id))`` with 1-based ranks.

    An id missing from a ranking contributes nothing for it. Ties are broken by the best rank any list gave the id,
    then by id, so the result is deterministic. ``k=60`` is the conventional constant; it is a parameter, not a tuned value.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking, w in zip(rankings, weights):
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # a repeated id in one list counts once, at its first (best) rank
                continue
            seen.add(doc_id)
            score[doc_id] = score.get(doc_id, 0.0) + w / (k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    return sorted(score.items(), key=lambda p: (-p[1], best[p[0]], p[0]))
