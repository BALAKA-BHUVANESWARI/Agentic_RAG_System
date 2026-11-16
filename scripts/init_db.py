"""Create the SQLite database. Usage: python scripts/init_db.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from copilot.config import get_settings  # noqa: E402
from copilot.db.schema import SCHEMA_VERSION, init_db, list_tables  # noqa: E402
from copilot.logging_config import get_logger, setup_logging  # noqa: E402


def main() -> None:
    """Initialise the database at SQLITE_PATH and print the resulting tables."""
    settings = get_settings()
    setup_logging()
    log = get_logger("init_db")
    init_db(settings.sqlite_path)
    tables = sorted(list_tables(settings.sqlite_path))
    log.info("Database ready", extra={"path": str(settings.sqlite_path), "schema_version": SCHEMA_VERSION})
    print(f"OK: {settings.sqlite_path}")
    print("Tables:", ", ".join(tables))


if __name__ == "__main__":
    main()
