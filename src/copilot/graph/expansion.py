"""Optional query expansion before the FIRST private retrieval (experiment C, flag ``RETRIEVAL_QUERY_EXPANSION``).

One extra LLM call writes up to ``n`` short paraphrases of the question that use the technical terms an IT document would use.
The paraphrases are only embedded for retrieval: they are never shown to the grader or the generator, never sent to web search and
never stored as the user's question. Any failure returns no paraphrases, so the request continues exactly as without the flag.
"""
from __future__ import annotations

from typing import Any

from copilot.generation.evidence import neutralize_delimiters
from copilot.graph.schemas import QueryExpansion
from copilot.llm.client import LLMUnavailableError, LLMUsage, StructuredOutputError
from copilot.llm.prompts import EXPANDER_SYSTEM_PROMPT
from copilot.logging_config import get_logger
from copilot.search.sanitizer import has_blocking_secret

log = get_logger(__name__)
MAX_PARAPHRASE_CHARS = 200


def clean_paraphrases(raw: list[str], question: str, n: int) -> list[str]:
    """At most ``n`` non-empty paraphrases that differ from the question and from each other (case-insensitive), trimmed."""
    seen = {" ".join(question.lower().split())}
    out: list[str] = []
    for text in raw:
        cleaned = " ".join(str(text).split())[:MAX_PARAPHRASE_CHARS]
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
        if len(out) >= n:
            break
    return out


def expand_query(llm: Any, question: str, n: int) -> tuple[list[str], LLMUsage | None]:
    """Paraphrases for ``question`` (empty on any failure, when ``n`` is 0, or when the question holds a pasted secret)."""
    if n <= 0 or has_blocking_secret(question):
        return [], None
    prompt = (
        f"<question>\n{neutralize_delimiters(question.strip())}\n</question>\n"
        f"Number of paraphrases requested: {n}"
    )
    try:
        result, usage = llm.structured([{"role": "system", "content": EXPANDER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], QueryExpansion)
    except (LLMUnavailableError, StructuredOutputError) as exc:
        log.warning("Query expansion failed; retrieving with the original question only", extra={"error": type(exc).__name__})
        return [], None
    return clean_paraphrases(result.paraphrases, question, n), usage
