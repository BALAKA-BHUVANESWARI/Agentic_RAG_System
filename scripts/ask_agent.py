"""Ask the full agentic workflow and print the answer plus the real decision trace.

Usage:
    python scripts/ask_agent.py "How do I connect to the company VPN from home?"
    python scripts/ask_agent.py "What is the latest Microsoft Teams outage guidance?"
    python scripts/ask_agent.py "How does the service desk verify a caller?" --role admin
    python scripts/ask_agent.py "latest Teams outage guidance" --no-external
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.graph.runner import AgenticRAG  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402


def main() -> int:
    """Run one agentic request and print path, trace and answer."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--role", default=None)
    parser.add_argument("--no-external", action="store_true", help="Forbid public web search for this request")
    parser.add_argument("--verbose", action="store_true", help="Print each trace event's data")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    r = AgenticRAG().answer(args.question, args.role, user_id="cli", external_search_allowed=not args.no_external)
    print("=" * 72)
    print(r.answer.rendered)
    print("=" * 72)
    print("PATH:", " -> ".join(r.path))
    for ev in r.trace:
        print(f"  [{ev.duration_ms:8.1f} ms] {ev.node:<24} {ev.summary}" + (f"  => {ev.decision}" if ev.decision else ""))
        if args.verbose:
            for k, v in ev.data.items():
                print(f"{'':35}{k}: {v}")
    print(f"\nabstained={r.answer.abstained} reason={r.answer.abstention_reason} role={r.role}")
    print(f"rewrites={r.rewritten_queries} external_search={r.external_search_triggered} tavily_calls={r.tavily_calls}")
    if r.external_blocked_reason:
        print("external not used:", r.external_blocked_reason)
    print(f"tokens={r.llm_usage.total_tokens} llm_calls={r.llm_usage.calls} latency={r.latency_ms:.0f} ms request_id={r.request_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
