"""Evaluation endpoints (Phase 7b): admin-only, background runs, status, summary, cancel, resume. No network."""
from __future__ import annotations

import threading
import time

import pytest

from eval_helpers import StubAgent, StubBaseline

from api_helpers import ADM, EMP


def test_every_evaluation_endpoint_is_admin_only(eval_api):
    ns, _ = eval_api()
    for method, url in [("post", "/evaluation/run"), ("get", "/evaluation/runs"), ("get", "/evaluation/runs/x"),
                        ("get", "/evaluation/runs/x/summary"), ("post", "/evaluation/runs/x/cancel")]:
        assert getattr(ns.client, method)(url, headers=EMP).status_code == 403, url


def test_request_validation(eval_api):
    ns, _ = eval_api()
    post = lambda body: ns.client.post("/evaluation/run", json=body, headers=ADM)  # noqa: E731
    assert post({"mode": "bogus"}).status_code == 422
    assert post({"categories": ["not_a_category"]}).status_code == 422
    assert post({"limit": 0}).status_code == 422
    assert post({"categories": ["out_of_scope"]}).status_code == 400            # valid category, but no case in the tiny dataset
    assert post({"resume_run_id": "../x"}).status_code == 400
    assert post({"resume_run_id": "never-ran"}).status_code == 400
    # dashboard options: bad values are rejected by the schema, valid ones reach the runner and fail there for honest reasons
    assert post({"split": "bogus"}).status_code == 422
    assert post({"variants": ["bogus"]}).status_code == 422
    assert post({"variants": ["grade"], "mode": "agentic"}).status_code == 422          # `mode` or `variants`, not both
    assert post({"variants": ["grade"]}).status_code == 400                             # valid rung, but this runner has no agent factory
    assert post({"attacks_only": True}).status_code == 400                              # the tiny dataset has no adversarial question
    assert post({"split": "test"}).status_code == 400                                   # the real split file does not match the tiny dataset


def test_start_poll_and_read_results(eval_api):
    ns, service = eval_api()
    r = ns.client.post("/evaluation/run", json={"use_judge": False}, headers=ADM)
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    assert r.json()["status"] == "running" and r.json()["total"] == 6
    service.wait(20)

    st = ns.client.get(f"/evaluation/runs/{run_id}", headers=ADM).json()
    assert st["status"] == "completed" and st["done"] == st["total"] == 6 and st["metrics"]["agentic"]["overall_correct_rate"] == 1.0
    summary = ns.client.get(f"/evaluation/runs/{run_id}/summary", headers=ADM).json()
    assert set(summary["modes"]) == {"baseline", "agentic"} and summary["comparison"]["findings"]
    assert [x["run_id"] for x in ns.client.get("/evaluation/runs", headers=ADM).json()] == [run_id]
    assert service.status(run_id)["status"] == "completed"


def test_unknown_or_malformed_run_ids_are_404(eval_api):
    ns, _ = eval_api()
    for rid in ("nope", "bad.id", "x" * 80):
        assert ns.client.get(f"/evaluation/runs/{rid}", headers=ADM).status_code == 404
        assert ns.client.get(f"/evaluation/runs/{rid}/summary", headers=ADM).status_code == 404


def test_one_run_at_a_time_cancel_and_resume(eval_api):
    gate = threading.Event()
    agent = StubAgent(gate=gate)
    ns, service = eval_api(agent=agent)
    first = ns.client.post("/evaluation/run", json={"mode": "agentic", "use_judge": False}, headers=ADM)
    assert first.status_code == 202
    run_id = first.json()["run_id"]
    assert service.active_run_id() == run_id
    # The runner does a retrieval pass before the first case. Wait until the first case is really in flight
    # (blocked on the gate); otherwise a fast cancel can land before any case starts and done would be 0.
    deadline = time.monotonic() + 10
    while not agent.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert agent.calls, "the first case never started"

    busy = ns.client.post("/evaluation/run", json={}, headers=ADM)
    assert busy.status_code == 409 and run_id in busy.json()["detail"]
    assert ns.client.post("/evaluation/runs/other/cancel", headers=ADM).status_code == 409
    assert ns.client.post(f"/evaluation/runs/{run_id}/cancel", headers=ADM).json() == {"run_id": run_id, "cancel_requested": True}
    assert ns.client.get(f"/evaluation/runs/{run_id}", headers=ADM).json()["status"] == "cancelling"

    gate.set()                                  # let the in-flight case finish; the run then stops
    service.wait(20)
    st = ns.client.get(f"/evaluation/runs/{run_id}", headers=ADM).json()
    assert st["status"] == "cancelled" and st["done"] == 1 and st["total"] == 3
    assert ns.client.get(f"/evaluation/runs/{run_id}/summary", headers=ADM).json()["status"] == "cancelled"

    resumed = ns.client.post("/evaluation/run", json={"resume_run_id": run_id}, headers=ADM)
    assert resumed.status_code == 202 and resumed.json()["run_id"] == run_id
    service.wait(20)
    assert ns.client.get(f"/evaluation/runs/{run_id}", headers=ADM).json()["status"] == "completed"
    assert ns.client.get(f"/evaluation/runs/{run_id}", headers=ADM).json()["done"] == 3


def test_a_run_left_running_by_a_dead_process_is_reported_as_interrupted(eval_api):
    ns, service = eval_api()
    service._store.start_run("ghost", "both", "ds.json", 3, {})      # a row nobody is working on
    assert ns.client.get("/evaluation/runs/ghost", headers=ADM).json()["status"] == "interrupted"


def test_an_ablation_run_started_through_the_api_returns_the_ladder(eval_api):
    ns, service = eval_api(agent_factory=lambda overrides: StubAgent())
    r = ns.client.post("/evaluation/run", json={"variants": ["agentic", "baseline", "grade"], "use_judge": False}, headers=ADM)
    assert r.status_code == 202 and r.json()["total"] == 9                              # 3 questions x 3 rungs
    service.wait(20)
    summary = ns.client.get(f"/evaluation/runs/{r.json()['run_id']}/summary", headers=ADM).json()
    assert summary["options"]["variants"] == ["baseline", "grade", "agentic"]           # ladder order, whatever order was sent
    assert list(summary["modes"]) == ["baseline", "grade", "agentic"]
    assert summary["ablation"]["order"] == ["baseline", "grade", "agentic"] and summary["intervals"]["paired"]
