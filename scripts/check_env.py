"""Check that the local environment is ready. Usage: python scripts/check_env.py

Verifies Python version, required packages, API keys (presence only, values are
never printed) and the synthetic KB documents. Exits non-zero if anything required
is missing.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_PACKAGES = [
    "pydantic", "pydantic_settings", "dotenv", "langgraph", "langchain_core",
    "langchain_text_splitters", "groq", "pinecone", "tavily", "sentence_transformers",
    "pypdf", "docx", "fastapi", "uvicorn", "multipart", "streamlit", "requests", "httpx", "pytest",
]
DISCLAIMER = "SYNTHETIC DEMO DOCUMENT"


def main() -> int:
    """Run all checks and return a process exit code."""
    failures = 0

    ok = sys.version_info >= (3, 11)
    print(f"[{'OK' if ok else 'FAIL'}] Python {sys.version.split()[0]} (need >= 3.11)")
    failures += not ok

    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"[OK]   import {pkg}")
        except Exception as exc:  # noqa: BLE001 - report any import problem
            print(f"[FAIL] import {pkg}: {type(exc).__name__}")
            failures += 1

    try:
        from copilot.config import get_settings

        settings = get_settings()
        for name in ("groq", "pinecone", "tavily"):
            try:
                settings.require_secret(name)  # type: ignore[arg-type]
                print(f"[OK]   {name.upper()}_API_KEY is set")
            except RuntimeError:
                print(f"[WARN] {name.upper()}_API_KEY not set (needed from Phase 2/3 onward)")
        kb = settings.kb_documents_dir
        docs = sorted(p for p in kb.glob("*") if p.suffix.lower() in settings.allowed_extensions)
        bad = [p.name for p in docs if DISCLAIMER not in p.read_text(encoding="utf-8", errors="ignore")[:200]]
        ok = len(docs) > 0 and not bad
        print(f"[{'OK' if ok else 'FAIL'}] {len(docs)} KB documents in {kb}" + (f"; missing disclaimer: {bad}" if bad else ""))
        failures += not ok
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] settings could not load: {exc}")
        failures += 1

    print("\nAll required checks passed." if not failures else f"\n{failures} required check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
