"""Verify the Groq key and configured model. Usage: python scripts/check_groq.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.llm.client import GroqClient, LLMUnavailableError  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402


def main() -> int:
    """List available models, confirm the configured one exists, and make a tiny call."""
    setup_logging(level="WARNING")
    settings = get_settings()
    client = GroqClient(settings)
    try:
        models = client.list_models()
        print(f"Configured model: {settings.groq_model}")
        print(f"Available to your key ({len(models)}): {', '.join(models)}")
        if settings.groq_model not in models:
            print("\nFAIL: GROQ_MODEL is not available to your key. Set GROQ_MODEL in .env to one listed above.")
            return 1
        reply = client.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=200)
        print(f"\nOK: model replied {reply.text.strip()!r} in {reply.latency_ms:.0f} ms, {reply.usage.total_tokens} tokens")
        return 0
    except (LLMUnavailableError, RuntimeError) as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
