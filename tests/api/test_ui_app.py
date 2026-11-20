"""Headless Streamlit tests: the real UI code driving the real FastAPI app (fake LLM/search/index)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from copilot.ui.api_client import CopilotClient

APP = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app.py"


@pytest.fixture
def app(monkeypatch):
    """Factory: ``app(api_ns)`` -> AppTest wired to that API instance."""

    def _build(api_ns):
        monkeypatch.setattr("copilot.ui.runtime.make_client", lambda role, user: CopilotClient("http://testserver", role, user, http=api_ns.client))
        return AppTest.from_file(str(APP), default_timeout=60)

    return _build


def _texts(at):
    return [m.value for m in at.markdown] + [i.value for i in at.info] + [w.value for w in at.warning]


def test_initial_render_shows_demo_identity_warning(app, chat_api):
    at = app(chat_api).run()
    assert not at.exception
    assert any("NOT authentication" in w.value for w in at.sidebar.warning)
    assert at.sidebar.selectbox(key="role").value == "employee" and at.sidebar.radio(key="page").options == ["Chat"]


def test_asking_a_question_shows_answer_citations_and_trace(app, chat_api):
    at = app(chat_api).run()
    at.chat_input[0].set_value("How do I connect to the company VPN from home?").run()
    assert not at.exception
    assert any("SecureLink" in t for t in _texts(at))
    assert any("VPN_Connection_Guide.txt" in m.value for m in at.markdown)  # source details
    at.toggle(key="trace_1").set_value(True).run()  # message 0 = user, message 1 = assistant
    assert not at.exception
    titles = " ".join(e.label for e in at.expander)
    assert "Query classification" in titles and "Private evidence grading" in titles and "External search decision" in titles
    assert any(m.label == "Tavily calls" and m.value == "0" for m in at.metric)


def test_feedback_button_records_feedback(app, chat_api):
    at = app(chat_api).run()
    at.chat_input[0].set_value("How do I connect to the company VPN from home?").run()
    at.button(key="fb_positive_1").click().run()
    assert not at.exception and any("feedback recorded" in c.value for c in at.caption)
    rid = next(iter(at.session_state["feedback"]))
    assert chat_api.svc.feedback.for_request(rid)[0]["rating"] == "positive"


def test_backend_error_is_shown_not_crashed(app, chat_api):
    at = app(chat_api).run()
    at.sidebar.text_input(key="user_id").set_value("bad user!").run()
    at.chat_input[0].set_value("hello there").run()
    assert not at.exception and any("X-Demo-User" in e.value for e in at.error)


def test_admin_page_lists_all_documents_and_is_hidden_from_employees(app, api):
    ns = api(load_kb=True)
    at = app(ns).run()
    assert at.sidebar.radio(key="page").options == ["Chat"]
    at.sidebar.selectbox(key="role").select("admin").run()
    assert at.sidebar.radio(key="page").options == ["Chat", "Admin"]
    at.sidebar.radio(key="page").set_value("Admin").run()
    assert not at.exception
    rows = at.dataframe[0].value
    assert len(rows) == 11 and "Service_Desk_Runbook.txt" in set(rows["name"])


def test_admin_page_shows_the_superseded_policy_and_the_version_status_field(app, api):
    ns = api(load_kb=True)
    at = app(ns).run()
    at.sidebar.selectbox(key="role").select("admin").run()
    at.sidebar.radio(key="page").set_value("Admin").run()
    assert not at.exception
    rows = at.dataframe[0].value
    status = dict(zip(rows["name"], rows["version status"]))
    assert status["Password_Policy_Legacy_2022.txt"] == "Superseded by Password_Policy.txt" and status["Password_Policy.txt"] == "Current"
    assert "Version status" in [s.label for s in at.selectbox]
