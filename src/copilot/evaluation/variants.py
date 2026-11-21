"""Named system variants for the ablation study: what does each agent step buy?

The ladder adds one step at a time, so the change in a metric between two neighbouring rungs can be attributed to
the step that was added:

    baseline  retrieve -> generate (no agent at all)
    classify  + query classifier (out-of-scope questions skip retrieval); the LLM evidence grader is OFF
    grade     + LLM evidence grader (insufficient evidence abstains; nothing else can recover it)
    rewrite   + query rewriter / retrieval retry (bounded by MAX_REWRITE_RETRIES)
    external  + Tavily fallback (graded, private-first, same routing rules as the full agent)
    agentic   + final LLM grounding validator  == the real application

Each rung is the SAME graph with settings overrides (``agent_use_grader``, ``agent_use_validator``,
``max_rewrite_retries``, ``external_search_enabled``); there is no second implementation of the workflow to drift.
Overrides are written down here and saved with every run, so a result can always be traced to its configuration.

Known confounds (also stated in docs/ABLATION.md): the ``classify`` rung changes two things relative to the
baseline, the classifier's out-of-scope shortcut and the agent's own answer path; every rung shares one model,
one corpus, one retriever and one prompt set, but not one random seed (LLM sampling is not deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


@dataclass(frozen=True)
class Variant:
    """One named system configuration."""

    name: str
    label: str
    description: str
    kind: Literal["baseline", "agent"]
    overrides: dict[str, Any] = field(default_factory=dict)  # Settings fields that differ from the full agent


LADDER: tuple[Variant, ...] = (
    Variant("baseline", "Baseline RAG", "retrieve, then generate with citations; no agent steps", "baseline"),
    Variant(
        "classify", "+ classifier", "classifier and out-of-scope shortcut; every retrieved chunk is trusted; no retry, no web, no validator", "agent",
        {"agent_use_grader": False, "max_rewrite_retries": 0, "external_search_enabled": False, "agent_use_validator": False},
    ),
    Variant(
        "grade", "+ evidence grader", "adds the LLM evidence grader; weak evidence abstains; no retry, no web, no validator", "agent",
        {"max_rewrite_retries": 0, "external_search_enabled": False, "agent_use_validator": False},
    ),
    Variant(
        "rewrite", "+ query rewrite", "adds query rewriting and bounded retrieval retries; no web, no validator", "agent",
        {"external_search_enabled": False, "agent_use_validator": False},
    ),
    Variant("external", "+ external fallback", "adds graded Tavily fallback under the private-first rules; no validator", "agent", {"agent_use_validator": False}),
    Variant("agentic", "+ final validator (full agent)", "adds the final grounding validator: the real application", "agent"),
)

BY_NAME: dict[str, Variant] = {v.name: v for v in LADDER}
NAMES: tuple[str, ...] = tuple(BY_NAME)


def get_variant(name: str) -> Variant:
    """The variant called ``name`` (raises ``ValueError`` listing the valid names)."""
    try:
        return BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown variant {name!r}; valid: {', '.join(NAMES)}") from None


def normalise(names: Iterable[str]) -> list[str]:
    """Validate, drop duplicates and return the names in ladder order (so tables always read bottom-up)."""
    wanted = {n.strip() for n in names if n and n.strip()}
    for n in wanted:
        get_variant(n)
    return [n for n in NAMES if n in wanted]


def parse_variants(text: str) -> list[str]:
    """``"baseline,grade,agentic"`` or ``"all"`` -> ladder-ordered names."""
    if text.strip().lower() == "all":
        return list(NAMES)
    names = normalise(text.split(","))
    if not names:
        raise ValueError("no variants given")
    return names


def overrides_for(name: str) -> dict[str, Any]:
    """Settings overrides for a variant (empty for the baseline and the full agent)."""
    return dict(get_variant(name).overrides)


def is_agent(name: str) -> bool:
    """True for every variant that runs the LangGraph workflow (everything except the baseline)."""
    return get_variant(name).kind == "agent"
