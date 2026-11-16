"""Ask the baseline RAG (retrieve -> Groq answer with citations).

Usage:
    python scripts/ask_baseline.py "How do I connect to the company VPN from home?"
    python scripts/ask_baseline.py "How does the service desk verify a caller?" --role admin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.baseline.rag import BaselineRAG  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402


def main() -> int:
    """Run one baseline RAG request and print the answer plus diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--role", default=None, help="employee (default) or admin")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    resp = BaselineRAG().answer(args.question, args.role)
    a = resp.answer
    print("=" * 70)
    print(a.rendered)
    print("=" * 70)
    print(f"abstained={a.abstained} reason={a.abstention_reason} role={resp.role}")
    if resp.retrieval:
        print("retrieved:", ", ".join(f"{c.document_name}({c.score:.2f})" for c in resp.retrieval.chunks))
    if a.invalid_markers_removed or a.uncited_lines:
        print(f"citation warnings: invalid_markers_removed={a.invalid_markers_removed} uncited_lines={len(a.uncited_lines)}")
    if resp.error or a.error:
        print("error:", resp.error or a.error)
    print(f"tokens={a.usage.total_tokens} llm_calls={a.usage.calls} latency={resp.latency_ms:.0f} ms model={a.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
