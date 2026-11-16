"""Dependencies shared by all graph nodes (injected so tests can use fakes)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from copilot.audit.logger import AuditLogger
from copilot.config import Settings
from copilot.generation.answer import AnswerGenerator


@dataclass(frozen=True)
class GraphDeps:
    """Everything the nodes need. ``llm`` needs ``structured(messages, schema)``,
    ``retriever`` needs ``retrieve(query, role)``, ``search`` needs ``search(query)``."""

    settings: Settings
    llm: Any
    retriever: Any
    search: Any
    generator: AnswerGenerator
    audit: AuditLogger
