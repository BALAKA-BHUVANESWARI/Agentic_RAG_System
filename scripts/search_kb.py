"""Query the private knowledge base directly (no LLM). Tests retrieval and access control.

Usage:
    python scripts/search_kb.py "How do I connect to the VPN from home?"
    python scripts/search_kb.py "how do I verify a caller" --role admin
    python scripts/search_kb.py "password length" --category password --top-k 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.logging_config import setup_logging  # noqa: E402
from copilot.vectorstore.models import RetrievalFilters  # noqa: E402
from copilot.vectorstore.retriever import Retriever  # noqa: E402


def main() -> int:
    """Run one retrieval and print the ranked chunks."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--role", default=None, help="employee (default) or admin")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    setup_logging(level="WARNING")
    result = Retriever().retrieve(
        args.query, args.role, top_k=args.top_k, filters=RetrievalFilters(category=args.category)
    )
    print(f"role={result.role} status={result.status} returned={len(result.chunks)} "
          f"dropped_unauthorized={result.dropped_unauthorized} dropped_low_score={result.dropped_low_score} reranked={result.reranked}\n")
    for i, c in enumerate(result.chunks, start=1):
        snippet = " ".join(c.text.split())[:140]
        rr = f" rerank={c.rerank_score:.2f}" if c.rerank_score is not None else ""
        print(f"{i}. score={c.score:.3f}{rr} [{c.access_level}] {c.document_name} ({c.location})\n   {snippet}...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
