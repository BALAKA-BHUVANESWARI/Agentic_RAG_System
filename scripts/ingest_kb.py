"""Ingest the synthetic KB documents into SQLite + Pinecone.

Usage:
    python scripts/ingest_kb.py                # ingest data/kb_documents (skips unchanged)
    python scripts/ingest_kb.py --force        # re-embed and re-index everything
    python scripts/ingest_kb.py --dir path\\to\\folder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.db.schema import init_db  # noqa: E402
from copilot.ingestion.pipeline import IngestionPipeline  # noqa: E402
from copilot.logging_config import setup_logging  # noqa: E402


def main() -> int:
    """Run ingestion and print a summary table. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=None, help="Folder to ingest (default: KB_DOCUMENTS_DIR)")
    parser.add_argument("--force", action="store_true", help="Re-index even if content is unchanged")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(level="WARNING")  # keep the summary readable; use LOG_LEVEL=INFO in the API
    init_db(settings.sqlite_path)
    pipeline = IngestionPipeline(settings)
    pipeline._store.ensure_index()  # noqa: SLF001 - create index up front for a clear error

    results = pipeline.ingest_directory(args.dir, uploaded_by="ingest_kb", force=args.force)
    print(f"\n{'STATUS':<9} {'VER':<4} {'CHUNKS':<7} DOCUMENT")
    for r in results:
        print(f"{r.status:<9} {str(r.version or '-'):<4} {r.chunk_count:<7} {r.document_name}  {('- ' + r.message) if r.status != 'indexed' else ''}")
    failed = [r for r in results if r.status == "failed"]
    print(f"\n{len(results)} files: {sum(r.status == 'indexed' for r in results)} indexed, "
          f"{sum(r.status == 'skipped' for r in results)} skipped, {len(failed)} failed")
    print("Note: Pinecone is eventually consistent; wait a few seconds before searching.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
