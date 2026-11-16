"""Start the FastAPI backend locally.

Usage:  python scripts/run_api.py          (docs at http://127.0.0.1:8000/docs)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from copilot.config import get_settings  # noqa: E402


def main() -> int:
    """Run uvicorn with the app factory."""
    s = get_settings()
    print(f"Starting API on http://{s.api_host}:{s.api_port}  (docs: /docs). Loading the embedding model takes ~20 s.")
    print("The server keeps running until you press Ctrl+C. Use a SECOND terminal to call it.")
    uvicorn.run("copilot.api.main:create_app", factory=True, host=s.api_host, port=s.api_port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
