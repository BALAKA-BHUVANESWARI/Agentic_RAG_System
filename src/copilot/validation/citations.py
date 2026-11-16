"""Deterministic citation checks on answer text (pure functions, no network).

Citations in answers are numeric markers such as ``[1]``, ``[1][2]`` or ``[1, 2]``
that refer to the numbered evidence blocks given to the model. Source lines are
rendered by code from real evidence, never written by the LLM.
"""
from __future__ import annotations

import re

MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_MIN_CLAIM_LINE_CHARS = 25


def _numbers(group: str) -> list[int]:
    return [int(n) for n in re.split(r"\s*,\s*", group)]


def extract_markers(text: str) -> list[int]:
    """Citation numbers in order of first appearance, without duplicates."""
    seen: list[int] = []
    for match in MARKER_RE.finditer(text):
        for n in _numbers(match.group(1)):
            if n not in seen:
                seen.append(n)
    return seen


def strip_invalid_markers(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Remove markers that do not refer to real evidence.

    Returns ``(clean_text, removed_numbers)``. A marker group with some valid and
    some invalid numbers keeps only the valid ones.
    """
    removed: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        nums = _numbers(match.group(1))
        good = [n for n in nums if n in valid]
        removed.extend(n for n in nums if n not in valid)
        if len(good) == len(nums):
            return match.group(0)
        return "[" + ", ".join(map(str, good)) + "]" if good else ""

    cleaned = MARKER_RE.sub(_replace, text)
    if removed:  # tidy spacing left behind, but only when we actually changed something
        cleaned = re.sub(r"(?<=\S)[ \t]+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"(?<=\S)[ \t]{2,}", " ", cleaned)
    return cleaned, sorted(set(removed))


def uncited_lines(text: str, valid: set[int]) -> list[str]:
    """Lines that look like claims but carry no valid citation (a heuristic SIGNAL only).

    Short lines and lines ending with ':' (introductions such as "Steps:") are ignored.
    """
    flagged: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if len(line) < _MIN_CLAIM_LINE_CHARS or line.endswith(":"):
            continue
        if not any(n in valid for n in extract_markers(line)):
            flagged.append(line)
    return flagged
