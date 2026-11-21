"""Small, dependency-free statistics for the evaluation: how much a rate or a difference can be trusted.

With 93 questions one case moves a rate by about one point, so a bare percentage says little. Two tools:

* :func:`wilson_interval`: interval for a single rate (num out of den). Unlike the textbook normal interval it stays
  inside 0-1 and behaves for rates at 0% or 100%, which this project often has.
* :func:`paired_bootstrap`: interval for the DIFFERENCE between two systems that answered the SAME questions. It
  resamples the questions (not the systems), so the pairing is respected, and also counts the cases that flipped.

Everything is seeded, so the same run always prints the same interval. The intervals describe sampling noise over
*which questions were asked* only. They do not cover LLM sampling noise between repeated runs, and they assume the
questions are exchangeable draws from the same population, which a hand-written dataset only approximates.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Any, Sequence


def _z(confidence: float) -> float:
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson_interval(num: int, den: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Wilson score interval for ``num`` successes out of ``den`` (None when ``den`` is 0)."""
    if den <= 0:
        return None
    if not 0 <= num <= den:
        raise ValueError("num must be between 0 and den")
    z = _z(confidence)
    p = num / den
    denom = 1 + z * z / den
    centre = (p + z * z / (2 * den)) / denom
    half = z * math.sqrt(p * (1 - p) / den + z * z / (4 * den * den)) / denom
    lo = 0.0 if num == 0 else max(0.0, centre - half)  # exact bounds at the extremes (avoids 2.8e-17 style noise)
    hi = 1.0 if num == den else min(1.0, centre + half)
    return lo, hi


def paired_bootstrap(
    before: Sequence[float], after: Sequence[float], *, resamples: int = 2000, seed: int = 0, confidence: float = 0.95
) -> dict[str, Any]:
    """Difference ``mean(after) - mean(before)`` over the same cases, with a percentile bootstrap interval.

    ``gained`` counts cases that went from 0 to 1 and ``lost`` cases that went from 1 to 0 (values are 0/1 outcomes).
    ``excludes_zero`` is true when the interval does not contain 0. Raises ``ValueError`` on empty or unequal input.
    """
    if len(before) != len(after):
        raise ValueError("before and after must cover the same cases")
    n = len(before)
    if n == 0:
        raise ValueError("no cases to compare")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    diffs = [float(a) - float(b) for b, a in zip(before, after)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    lo = means[min(resamples - 1, max(0, int(math.floor(tail * resamples))))]
    hi = means[min(resamples - 1, max(0, int(math.ceil((1 - tail) * resamples)) - 1))]
    point = sum(diffs) / n
    return {
        "n": n, "diff": point, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
        "gained": sum(1 for b, a in zip(before, after) if a and not b), "lost": sum(1 for b, a in zip(before, after) if b and not a),
    }
