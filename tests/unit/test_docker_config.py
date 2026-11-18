"""Static checks on the Docker packaging (no Docker, no network): secrets stay out, ports stay local, names are real."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from copilot.config import PROJECT_ROOT, Settings

yaml = pytest.importorskip("yaml")

DOCKERFILE = (PROJECT_ROOT / "Dockerfile")
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"

# Variables that are not application settings (they configure Hugging Face inside the container).
NON_SETTINGS_ENV = {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
SECRET_LIKE = re.compile(r"\b(gsk_|tvly-|pcsk_|sk-)[A-Za-z0-9_\-]{8,}")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_the_docker_files_exist():
    assert DOCKERFILE.is_file() and COMPOSE.is_file() and DOCKERIGNORE.is_file()


def test_the_image_contains_no_secrets_and_runs_as_a_normal_user():
    text = DOCKERFILE.read_text(encoding="utf-8")
    code = [ln for ln in _lines(DOCKERFILE) if ln.upper().startswith(("ENV ", "ARG ", "COPY ", "ADD "))]
    assert not any("API_KEY" in ln.upper() for ln in code), "API keys must not be build arguments or environment defaults"
    assert not any(re.search(r"\.env\b(?!\.example)", ln) for ln in code if ln.upper().startswith(("COPY ", "ADD ")))
    assert not SECRET_LIKE.search(text)
    assert re.search(r"^USER app$", text, re.M) and "useradd" in text            # not root
    assert "download.pytorch.org/whl/cpu" in text                                # CPU-only PyTorch (no CUDA download)
    assert "SentenceTransformer(" in text                                         # embedding model baked in
    assert "COPY data/kb_documents" in text and "COPY data/eval/dataset.json" in text


def test_dockerignore_keeps_secrets_environments_and_runtime_data_out():
    lines = set(_lines(DOCKERIGNORE))
    assert {".env", ".venv/", ".git"} <= lines and "!.env.example" in lines
    assert {"data/sqlite/*", "data/uploads/*", "data/eval/results/*"} <= lines


def test_compose_has_the_services_and_no_secret_values(compose):
    assert {"api", "ui"} <= set(compose["services"])
    assert {"ingest", "eval", "tests"} <= set(compose["services"])
    assert not SECRET_LIKE.search(COMPOSE.read_text(encoding="utf-8"))
    for name, svc in compose["services"].items():
        assert svc["env_file"] == ".env", name                                   # keys arrive at run time, from the user's .env
        assert not any("API_KEY" in key.upper() for key in svc["environment"]), name


def test_ports_are_published_on_loopback_only(compose):
    """The API trusts an unverified role header (demo auth), so it must not be reachable from the network."""
    for name in ("api", "ui"):
        for port in compose["services"][name]["ports"]:
            assert str(port).startswith("127.0.0.1:"), f"{name}: {port}"


def test_containers_talk_to_each_other_and_to_the_right_paths(compose):
    api_env, ui_env = compose["services"]["api"]["environment"], compose["services"]["ui"]["environment"]
    assert api_env["API_HOST"] == "0.0.0.0"                                       # reachable from the ui container
    assert ui_env["API_BASE_URL"] == "http://api:8000"
    assert api_env["API_PORT"] == "8000"
    assert api_env["SQLITE_PATH"].startswith("/app/data/") and api_env["UPLOADS_DIR"].startswith("/app/data/")
    assert compose["services"]["ui"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert "frontend/streamlit_app.py" in compose["services"]["ui"]["command"]


def test_every_container_setting_is_a_real_setting(compose):
    """Catches typos: a misspelt variable would be silently ignored by the application."""
    known = {name.upper() for name in Settings.model_fields}
    for name, svc in compose["services"].items():
        unknown = {k for k in svc["environment"] if k not in NON_SETTINGS_ENV and k not in known}
        assert not unknown, f"{name}: not Settings fields: {sorted(unknown)}"


def test_state_lives_in_named_volumes_shared_by_the_api_and_the_tools(compose):
    assert set(compose["volumes"]) == {"copilot-sqlite", "copilot-uploads", "copilot-eval-results"}
    api_volumes = set(compose["services"]["api"]["volumes"])
    for name in ("ingest", "eval", "tests"):
        assert set(compose["services"][name]["volumes"]) == api_volumes, name   # one database, one set of uploads and results
    assert all(not v.startswith(("./", "/")) for v in api_volumes)               # named volumes, not host bind mounts


def test_tools_are_opt_in_and_do_not_start_with_the_stack(compose):
    for name in ("ingest", "eval", "tests"):
        assert compose["services"][name]["profiles"] == ["tools"], name
        assert compose["services"][name]["restart"] == "no"
    assert compose["services"]["eval"]["entrypoint"] == ["python", "scripts/run_eval.py"]
    assert compose["services"]["eval"]["command"] == ["--validate-only"]         # the default does not spend provider tokens
