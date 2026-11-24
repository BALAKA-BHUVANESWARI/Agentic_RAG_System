"""Pre-flight checks before a token-spending evaluation run: each rule fails for the right reason and passes for the right one."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from copilot.evaluation import preflight as pf
from copilot.evaluation.preflight import Check

FRESH = Path(__file__).resolve().parents[2] / "data" / "eval" / "fresh_dataset.json"
CORE = ["MFA_Guidelines.txt", "Password_Policy.txt", "VPN_Connection_Guide.txt"]

GOOD_SETTINGS = {
    "rerank_enabled": False, "retrieval_max_chunks_per_doc": None, "agent_use_grader": True, "agent_use_validator": True,
    "external_search_enabled": True, "has_groq_key": True, "has_pinecone_key": True, "has_tavily_key": True,
    "llm_model": "model-x", "retrieval_top_k": 5, "retrieval_min_score": 0.25, "max_rewrite_retries": 2,
}


def by_name(checks):
    return {c.name: c for c in checks}


def rows(**overrides):
    base = {"MFA_Guidelines.txt": ("indexed", "current", 2), "Password_Policy.txt": ("indexed", "current", 3), "VPN_Connection_Guide.txt": ("indexed", "current", 4)}
    base.update(overrides)
    return [(name, *vals) for name, vals in base.items()]


# ------------------------------------------------------------------------------ settings
def test_good_settings_all_pass_and_the_values_in_use_are_shown():
    checks = pf.check_settings(GOOD_SETTINGS)
    assert {c.status for c in checks} == {"ok"}
    assert "model model-x, top-k 5, minimum retrieval score 0.25, max rewrite retries 2" in by_name(checks)["settings in use"].detail


@pytest.mark.parametrize("change,name,status,fragment", [
    ({"rerank_enabled": True}, "reranker off", "fail", "RERANK_ENABLED is true"),
    ({"retrieval_max_chunks_per_doc": 3}, "no per-document chunk cap", "fail", "is 3"),
    ({"agent_use_grader": False}, "full agent", "fail", "ablation rung"),
    ({"agent_use_validator": False}, "full agent", "fail", "ablation rung"),
    ({"external_label_check_in_code": True}, "experiment flags", "warn", "A external_label_check_in_code"),
    ({"partial_private_evidence": True}, "experiment flags", "warn", "B partial_private_evidence"),
    ({"external_search_enabled": False}, "web search enabled", "warn", "can only abstain"),
    ({"has_groq_key": False}, "Groq key", "fail", "GROQ_API_KEY is missing"),
    ({"has_pinecone_key": False}, "Pinecone key", "fail", "PINECONE_API_KEY is missing"),
    ({"has_tavily_key": False}, "Tavily key", "warn", "web search is enabled"),
])
def test_each_bad_setting_is_flagged_by_name(change, name, status, fragment):
    c = by_name(pf.check_settings({**GOOD_SETTINGS, **change}))[name]
    assert c.status == status and fragment in c.detail


def test_missing_tavily_key_is_fine_when_web_search_is_off():
    c = by_name(pf.check_settings({**GOOD_SETTINGS, "has_tavily_key": False, "external_search_enabled": False}))["Tavily key"]
    assert c.status == "ok"


# ------------------------------------------------------------------------------ dataset
def test_dataset_check_reports_the_first_problems():
    assert pf.check_dataset("d.json", 93, []).status == "ok"
    bad = pf.check_dataset("d.json", 93, ["a", "b", "c", "d"])
    assert bad.status == "fail" and "4 problem(s)" in bad.detail and bad.detail.endswith("a; b; c")      # only the first three are listed


def test_fingerprint_states():
    assert pf.check_fingerprint("abc", None).status == "skip"
    assert pf.check_fingerprint("abc", "abc").status == "ok"
    assert pf.check_fingerprint("abc", "xyz").status == "fail"


def test_fingerprint_is_the_canonical_hash_and_the_sidecar_sits_next_to_the_file(tmp_path):
    ds = tmp_path / "block.json"
    ds.write_text(json.dumps({"cases": [{"b": 1, "a": "é"}, {"id": 2}]}), encoding="utf-8")
    first = pf.cases_fingerprint(ds)
    ds.write_text(json.dumps({"cases": [{"a": "é", "b": 1}, {"id": 2}], "note": "other keys are ignored"}), encoding="utf-8")
    assert pf.cases_fingerprint(ds) == first                      # key order and non-case content do not matter
    assert pf.expected_fingerprint(ds) is None
    (tmp_path / "block.sha256").write_text(first + "\n", encoding="utf-8")
    assert pf.expected_fingerprint(ds) == first
    ds.write_text(json.dumps({"cases": [{"a": "é", "b": 2}, {"id": 2}]}), encoding="utf-8")
    assert pf.cases_fingerprint(ds) != first                      # an edited label changes it


@pytest.mark.skipif(not FRESH.exists(), reason="the fresh block is not part of the Docker image; checked on the host")
def test_the_real_fresh_block_matches_its_committed_fingerprint():
    assert pf.check_fingerprint(pf.cases_fingerprint(FRESH), pf.expected_fingerprint(FRESH)).status == "ok"


# ------------------------------------------------------------------------------ split
def test_split_without_a_file():
    assert pf.check_split("all", False, [], ["D01"], []).status == "ok"
    c = pf.check_split("test", False, [], ["D01"], [])
    assert c.status == "fail" and "needs the split file" in c.detail


def test_split_file_that_covers_none_of_the_dataset_is_the_fresh_block_trap():
    warn = pf.check_split("all", True, [], ["F01", "F02"], ["D01", "D02"])
    assert warn.status == "warn" and "EVAL_SPLIT_PATH" in warn.detail
    assert pf.check_split("dev", True, [], ["F01"], ["D01"]).status == "fail"


def test_split_problems_and_the_good_case():
    assert pf.check_split("test", True, ["ids in both dev and test: ['D01']"], ["D01"], ["D01"]).status == "fail"
    ok = pf.check_split("dev", True, [], ["D01", "D02", "F01"], ["D01", "D02"])
    assert ok.status == "ok" and "2 of 3 cases assigned" in ok.detail and "--split dev" in ok.detail


# ------------------------------------------------------------------------------ documents and index
def test_documents_exactly_the_core_set_all_indexed():
    checks = by_name(pf.check_documents(rows(**{"Password_Policy.txt": ("indexed", "superseded", 3)}), CORE))
    assert checks["SQLite document set"].status == "ok" and "exactly the 3 core documents" in checks["SQLite document set"].detail
    assert checks["SQLite indexing status"].status == "ok" and "1 marked superseded" in checks["SQLite indexing status"].detail


def test_extra_missing_and_unindexed_documents_fail():
    extra = by_name(pf.check_documents(rows(**{"Demo_Upload.txt": ("indexed", "current", 1)}), CORE))["SQLite document set"]
    assert extra.status == "fail" and "extra: Demo_Upload.txt" in extra.detail and "Admin > Documents" in extra.detail
    missing = by_name(pf.check_documents([r for r in rows() if r[0] != "MFA_Guidelines.txt"], CORE))["SQLite document set"]
    assert missing.status == "fail" and "missing: MFA_Guidelines.txt" in missing.detail
    failed = by_name(pf.check_documents(rows(**{"VPN_Connection_Guide.txt": ("failed", "current", 0)}), CORE))["SQLite indexing status"]
    assert failed.status == "fail" and "VPN_Connection_Guide.txt (failed)" in failed.detail


def test_index_matches_sqlite():
    counts = {"MFA_Guidelines.txt": 2, "Password_Policy.txt": 3, "VPN_Connection_Guide.txt": 4}
    checks = by_name(pf.check_index(counts, None, rows(), CORE))
    assert checks["Pinecone document set"].status == "ok" and "9 chunks" in checks["Pinecone document set"].detail
    assert checks["Pinecone chunk counts"].status == "ok"


def test_index_with_stale_vectors_extra_or_missing_documents():
    stale = by_name(pf.check_index({"MFA_Guidelines.txt": 2, "Password_Policy.txt": 5, "VPN_Connection_Guide.txt": 4}, None, rows(), CORE))["Pinecone chunk counts"]
    assert stale.status == "fail" and "Password_Policy.txt (index 5, SQLite 3)" in stale.detail
    odd = by_name(pf.check_index({"MFA_Guidelines.txt": 2, "Other.txt": 1}, None, rows(), CORE))["Pinecone document set"]
    assert odd.status == "fail" and "missing from the index: Password_Policy.txt, VPN_Connection_Guide.txt" in odd.detail and "extra in the index: Other.txt" in odd.detail


def test_index_skipped_or_unreadable():
    assert pf.check_index(None, None, rows(), CORE)[0].status == "skip"
    err = pf.check_index(None, "ConnectionError", rows(), CORE)
    assert len(err) == 1 and err[0].status == "fail" and "ConnectionError" in err[0].detail


# ------------------------------------------------------------------------------ settings against a saved run
SAVED = {"llm_model": "model-x", "rerank_enabled": False, "retrieval_top_k": 5, "eval_bootstrap_seed": 7}


def test_identical_settings_pass_and_unrecorded_keys_are_not_compared():
    current = {**SAVED, "retrieval_max_chunks_per_doc": None}      # a key the saved run never recorded
    c = pf.compare_settings(SAVED, current, label="settings vs fresh-1", consequence="x")
    assert c.status == "ok" and "all 4 comparable settings are identical" in c.detail


def test_a_changed_setting_fails_and_shows_both_values():
    c = pf.compare_settings(SAVED, {**SAVED, "rerank_enabled": True}, label="settings vs fresh-1", consequence="restore them")
    assert c.status == "fail" and "rerank_enabled False -> True" in c.detail and "restore them" in c.detail


def test_interval_only_differences_are_a_warning():
    c = pf.compare_settings(SAVED, {**SAVED, "eval_bootstrap_seed": 99}, label="l", consequence="x")
    assert c.status == "warn" and "eval_bootstrap_seed" in c.detail


def test_resume_target_states():
    assert pf.check_resumable("fresh-1", False, None).status == "fail"
    done = pf.check_resumable("fresh-1", True, "completed")
    assert done.status == "fail" and "already completed" in done.detail
    assert pf.check_resumable("fresh-1", True, "stopped").status == "ok"
    assert "started without a summary" in pf.check_resumable("fresh-1", True, None).detail


# ------------------------------------------------------------------------------ reading what exists
def test_read_documents_returns_rows_in_name_order(tmp_path):
    db = tmp_path / "app.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE documents (document_name TEXT, status TEXT, lifecycle TEXT, chunk_count INTEGER)")
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?)", [("b.txt", "indexed", "current", 3), ("a.txt", "failed", "superseded", 0)])
    conn.commit()
    conn.close()
    assert pf.read_documents(db) == [("a.txt", "failed", "superseded", 0), ("b.txt", "indexed", "current", 3)]


def test_read_documents_returns_none_for_a_missing_or_unusable_database(tmp_path):
    assert pf.read_documents(tmp_path / "nope.db") is None
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()                                  # no documents table
    assert pf.read_documents(empty) is None


def test_saved_run_prefers_config_settings_and_reads_status_from_the_summary(tmp_path):
    folder = tmp_path / "fresh-1"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"settings": {"llm_model": "from-config"}}), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({"status": "stopped", "settings": {"llm_model": "from-summary"}}), encoding="utf-8")
    assert pf.saved_run(tmp_path, "fresh-1") == (True, {"llm_model": "from-config"}, "stopped")
    (folder / "config.json").unlink()
    assert pf.saved_run(tmp_path, "fresh-1") == (False, {"llm_model": "from-summary"}, "stopped")
    assert pf.saved_run(tmp_path, "nothing") == (False, {}, None)


# ------------------------------------------------------------------------------ report
def test_render_and_verdict():
    checks = [Check("a", "ok", "fine"), Check("b", "warn", "careful"), Check("c", "skip", "not looked at")]
    assert pf.render(checks).splitlines() == ["[ OK ] a: fine", "[WARN] b: careful", "[SKIP] c: not looked at"]
    assert pf.verdict(checks) == (True, "READY: 1 check(s) passed, 1 warning(s), 1 skipped.")
    ready, sentence = pf.verdict(checks + [Check("d", "fail", "broken")])
    assert not ready and sentence == "NOT READY: 1 check(s) failed. Fix them before spending tokens."
