"""Deterministic text matching used by the evaluation (no LLM involved).

Text is lower-cased and every run of non-alphanumeric characters becomes one space, so
``SL-401``, ``sl 401`` and ``vpn.acme-demo.example`` compare equal to their spoken forms. A keyword
matches only on word boundaries ("5" does not match "15").
"""
from __future__ import annotations

import re
from typing import Iterable

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Normalised text padded with spaces so substring tests respect word boundaries."""
    return " " + _NON_ALNUM.sub(" ", text.lower()).strip() + " "


def contains(normalized_text: str, keyword: str) -> bool:
    """True if ``keyword`` occurs in already-normalised text."""
    return normalize(keyword) in normalized_text


def all_keywords_present(normalized_text: str, keywords: Iterable[str]) -> bool:
    """True if every keyword occurs."""
    return all(contains(normalized_text, k) for k in keywords)


def forbidden_hits(text: str, keywords: Iterable[str]) -> list[str]:
    """Keywords from ``keywords`` that appear in ``text`` (used to detect leaks of restricted content)."""
    norm = normalize(text)
    return [k for k in keywords if contains(norm, k)]
