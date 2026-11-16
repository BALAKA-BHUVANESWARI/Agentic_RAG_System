"""Integration tests need real API keys and an ingested KB; they skip cleanly otherwise."""
from __future__ import annotations

import pytest

from copilot.config import get_settings

pytestmark = pytest.mark.integration


def _require(*names: str) -> None:
    settings = get_settings()
    for name in names:
        try:
            settings.require_secret(name)  # type: ignore[arg-type]
        except RuntimeError:
            pytest.skip(f"{name.upper()}_API_KEY not set")


@pytest.fixture
def need_groq():
    _require("groq")


@pytest.fixture
def need_groq_and_pinecone():
    _require("groq", "pinecone")
