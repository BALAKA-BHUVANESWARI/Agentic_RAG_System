"""System-level facts recorded with every run: ingestion success, settings snapshot, dataset fingerprint."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from copilot.config import Settings
from copilot.db.connection import connect
from copilot.evaluation.aggregate import rate


def file_sha256(path: Path | str) -> str:
    """SHA-256 of a file (identifies exactly which dataset a run used)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ingestion_stats(db_path: Path | str) -> dict[str, Any] | None:
    """Ingestion success from the SQLite ingestion history (None if the tables are unavailable).

    ``attempts`` counts every ingestion attempt ever recorded, so a failed upload that was fixed later still lowers it.
    ``documents`` is the CURRENT state of each document.
    """
    try:
        with connect(db_path) as conn:
            attempts = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) AS n FROM ingestion_records GROUP BY status")}
            docs = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) AS n FROM documents GROUP BY status")}
    except sqlite3.Error:
        return None
    return {
        "attempts": rate(attempts.get("indexed", 0), sum(attempts.values())),
        "documents": rate(docs.get("indexed", 0), sum(docs.values())),
        "attempts_by_status": attempts, "documents_by_status": docs,
    }


def settings_snapshot(s: Settings) -> dict[str, Any]:
    """Non-secret configuration that influences results (so runs can be compared honestly)."""
    return {
        "llm_model": s.groq_model, "reasoning_effort": s.groq_reasoning_effort, "temperature": s.groq_temperature, "top_p": s.groq_top_p,
        "rerank_enabled": s.rerank_enabled, "rerank_model": s.rerank_model_name if s.rerank_enabled else None,
        "rerank_candidates": s.rerank_candidates if s.rerank_enabled else None,
        "embedding_model": s.embedding_model_name, "retrieval_top_k": s.retrieval_top_k, "retrieval_min_score": s.retrieval_min_score,
        "private_evidence_threshold": s.private_evidence_threshold, "external_evidence_threshold": s.external_evidence_threshold,
        "max_rewrite_retries": s.max_rewrite_retries, "max_validation_revisions": s.max_validation_revisions,
        "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, "external_search_enabled": s.external_search_enabled,
        "tavily_search_depth": s.tavily_search_depth, "eval_ks": s.eval_ks, "eval_retrieval_pool": s.eval_retrieval_pool,
        "eval_answer_threshold": s.eval_answer_threshold, "eval_faithfulness_threshold": s.eval_faithfulness_threshold,
    }
