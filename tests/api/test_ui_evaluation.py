"""Evaluation dashboard: the real Streamlit code and CopilotClient driving the real FastAPI app (stub systems, no network)."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from copilot.ui.api_client import ApiError, CopilotClient
from eval_helpers import StubAgent

APP = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"


def client_for(ns, role="admin", user="root"):
    return CopilotClient("http://testserver", role, user, http=ns.client)


@pytest.fixture
def admin_app(monkeypatch):
    """Factory: ``admin_app(ns)`` -> AppTest on the Admin page, wired to that API instance."""

    def _build(ns):
        monkeypatch.setattr("copilot.ui.runtime.make_client", lambda role, user: CopilotClient("http://testserver", role, user, http=ns.client))
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        at.sidebar.selectbox(key="role").select("admin").run()
        at.sidebar.radio(key="page").set_value("Admin").run()
        assert not at.exception
        return at

    return _build


def _messages(at) -> str:
    return " ".join([m.value for m in at.markdown] + [i.value for i in at.info] + [w.value for w in at.warning] + [e.value for e in at.error] + [s.value for s in at.success] + [c.value for c in at.caption])


# ------------------------------------------------------------------------------ client
def test_client_runs_an_evaluation_end_to_end(eval_api):
    ns, service = eval_api()
    c = client_for(ns)
    info = c.start_evaluation(use_judge=False)
    assert info["status"] == "running" and info["total"] == 6
    service.wait(20)
    run_id = info["run_id"]
    assert c.evaluation_status(run_id)["status"] == "completed"
    assert [r["run_id"] for r in c.list_evaluation_runs()] == [run_id]
    assert set(c.evaluation_summary(run_id)["modes"]) == {"baseline", "agentic"}


def test_client_surfaces_busy_cancel_and_missing_runs_as_readable_errors(eval_api):
    gate = threading.Event()
    ns, service = eval_api(agent=StubAgent(gate=gate))
    c = client_for(ns)
    run_id = c.start_evaluation(mode="agentic", use_judge=False)["run_id"]
    with pytest.raises(ApiError) as busy:
        c.start_evaluation(use_judge=False)
    assert busy.value.status_code == 409 and run_id in busy.value.detail
    assert c.cancel_evaluation(run_id)["cancel_requested"] is True
    gate.set()
    service.wait(20)
    assert c.evaluation_status(run_id)["status"] == "cancelled"
    with pytest.raises(ApiError) as missing:
        c.evaluation_summary("never-ran")
    assert missing.value.status_code == 404
    with pytest.raises(ApiError) as bad:
        c.start_evaluation(categories=["bogus"])
    assert bad.value.status_code == 422 and "categories" in bad.value.detail


def test_employees_cannot_use_the_evaluation_endpoints(eval_api):
    ns, _ = eval_api()
    with pytest.raises(ApiError) as exc:
        client_for(ns, role="employee", user="alice").list_evaluation_runs()
    assert exc.value.status_code == 403


# ------------------------------------------------------------------------------ dashboard
def test_evaluation_tab_without_a_backend_suite_explains_it(admin_app, api):
    at = admin_app(api(load_kb=False))
    assert "not installed" in _messages(at)
    assert not [b for b in at.button if b.key == "eval_start"]


def test_empty_state_offers_a_small_first_run(admin_app, eval_api):
    ns, _ = eval_api()
    at = admin_app(ns)
    assert "No evaluation runs yet" in _messages(at)
    assert at.number_input(key="eval_per_category").value == 1 and at.checkbox(key="eval_judge").value is False     # cheap by default
    assert at.button(key="eval_start").disabled is False


def test_starting_a_run_shows_its_results_and_comparison_honestly(admin_app, eval_api):
    ns, service = eval_api()
    at = admin_app(ns)
    at.session_state["eval_pause_refresh"] = True          # the test drives refreshes itself
    at.button(key="eval_start").click().run()
    assert not at.exception
    service.wait(20)
    at.run()
    assert not at.exception
    text = _messages(at)
    assert "Completed" in text and "Only 3 labelled questions per system" in text           # small-sample warning is always shown
    assert "Not enough evidence to conclude which system is better" in text               # never claims a winner on 3 questions
    assert any(d.value.shape[0] >= 1 for d in at.dataframe)
    labels = " ".join(t.label for t in at.tabs)
    for name in ("Overview", "Retrieval", "Answer quality", "Routing & external search", "Latency & cost", "Baseline vs agentic", "Failed cases"):
        assert name in labels


def test_a_second_start_while_busy_is_blocked_and_cancel_works(admin_app, eval_api):
    gate = threading.Event()
    ns, service = eval_api(agent=StubAgent(gate=gate))
    at = admin_app(ns)
    at.checkbox(key="eval_judge").set_value(False)
    at.selectbox(key="eval_mode").select("agentic").run()
    at.session_state["eval_pause_refresh"] = True          # the test drives refreshes itself
    at.button(key="eval_start").click().run()
    assert at.button(key="eval_start").disabled is True and "Run in progress" in _messages(at)
    at.button(key="eval_cancel").click().run()
    gate.set()
    service.wait(20)
    at.run()
    assert not at.exception and "Cancelled" in _messages(at)
    assert at.button(key="eval_resume").disabled is False          # a cancelled run can be resumed from the page


def test_resuming_from_the_page_finishes_the_run(admin_app, eval_api):
    gate = threading.Event()
    ns, service = eval_api(agent=StubAgent(gate=gate))
    c = client_for(ns)
    run_id = c.start_evaluation(mode="agentic", use_judge=False)["run_id"]
    c.cancel_evaluation(run_id)
    gate.set()
    service.wait(20)
    at = admin_app(ns)
    at.session_state["eval_pause_refresh"] = True
    at.button(key="eval_resume").click().run()
    service.wait(20)
    at.run()
    assert not at.exception and c.evaluation_status(run_id)["status"] == "completed" and "Completed" in _messages(at)


def test_failed_cases_are_listed_with_their_answers(admin_app, eval_api):
    ns, service = eval_api(agent=StubAgent(lambda q: RuntimeError("provider down")))
    client_for(ns).start_evaluation(mode="agentic", use_judge=False)
    service.wait(20)
    at = admin_app(ns)
    text = _messages(at)
    assert not at.exception and "Stopped" in text and "failed or incorrect case" in text          # 3 errors in a row stop the run
    assert any("agentic" in e.label and "error" in e.label for e in at.expander)
