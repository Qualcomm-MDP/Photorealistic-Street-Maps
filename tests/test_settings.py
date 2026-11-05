import json
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

import api.app
import api.settings
from api.app import create_app
from common import PipelineChain

TOKEN = "MLY|1234567890|savedtokenabcd"
ENV_TOKEN = "MLY|9876543210|envtokenwxyz"
BBOX = {"west": -83.71978, "south": 42.29025, "east": -83.71205, "north": 42.29422}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    # Don't read a developer's own .env, and put the token variable back after
    # each test (the settings store writes to os.environ).
    monkeypatch.setattr(api.app, "load_app_env", lambda: None)
    monkeypatch.delenv("MAPILLARY_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    # Env files live in the test's folder, never the repo. An (empty) .env
    # exists unless a test removes it to try the "no env file" case.
    monkeypatch.setenv("PSM_ENV_DIR", str(tmp_path))
    (tmp_path / ".env").write_text("")


def make_client(tmp_path, factory=None):
    kwargs = {"pipeline_factory": factory} if factory else {}
    return TestClient(create_app(data_dir=tmp_path, serve_web=False, **kwargs))


def test_no_token_anywhere(tmp_path):
    settings = make_client(tmp_path).get("/api/settings").json()

    assert settings["mapillary"] == {
        "set": False,
        "source": None,
        "hint": None,
        "env_available": False,
        "env_file": ".env",
        "env_file_exists": True,
    }
    assert settings["defaults"] == {
        "poses": "colmap",
        "remove_obstructions": True,
        "fill_obstructions": True,
        "seg_model": "large",
        "device": "auto",
    }


def test_token_from_env_names_the_app_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", ENV_TOKEN)
    monkeypatch.setenv("APP_ENV", "development")
    (tmp_path / ".env.development").write_text(f"MAPILLARY_ACCESS_TOKEN={ENV_TOKEN}\n")

    resp = make_client(tmp_path).get("/api/settings")

    assert resp.json()["mapillary"] == {
        "set": True,
        "source": "env",
        "hint": "…wxyz",
        "env_available": True,
        "env_file": ".env.development",
        "env_file_exists": True,
    }
    # The status never includes the token; only its own endpoint does.
    assert ENV_TOKEN not in resp.text


def test_token_endpoint_returns_the_current_token_uncached(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", ENV_TOKEN)
    client = make_client(tmp_path)

    env = client.get("/api/settings/mapillary/token")
    assert env.json() == {"token": ENV_TOKEN}
    assert env.headers["cache-control"] == "no-store"

    client.put("/api/settings", json={"mapillary_token": TOKEN})
    assert client.get("/api/settings/mapillary/token").json() == {"token": TOKEN}


def test_token_endpoint_without_a_token(tmp_path):
    resp = make_client(tmp_path).get("/api/settings/mapillary/token")

    assert resp.json() == {"token": None}


@pytest.mark.parametrize(
    "app_env, expected",
    [
        (None, ".env"),
        ("", ".env"),  # docker compose passes APP_ENV="" when it isn't set
        ("development", ".env.development"),
        ("production", ".env.production"),
    ],
)
def test_app_env_file_follows_app_env(monkeypatch, app_env, expected):
    from common.env import app_env_file

    if app_env is not None:
        monkeypatch.setenv("APP_ENV", app_env)

    assert app_env_file() == expected


def test_saved_token_is_used_kept_private_and_survives_restart(tmp_path):
    client = make_client(tmp_path)

    resp = client.put("/api/settings", json={"mapillary_token": f"  {TOKEN} "})

    assert resp.status_code == 200
    assert TOKEN not in resp.text
    assert resp.json()["mapillary"]["source"] == "saved"
    assert os.environ["MAPILLARY_ACCESS_TOKEN"] == TOKEN
    saved = tmp_path / "settings.json"
    assert json.loads(saved.read_text())["mapillary_token"] == TOKEN
    if sys.platform != "win32":
        assert saved.stat().st_mode & 0o777 == 0o600

    restarted = make_client(tmp_path).get("/api/settings").json()
    assert restarted["mapillary"]["source"] == "saved"
    assert restarted["mapillary"]["hint"] == "…abcd"


def test_saved_token_wins_over_env_and_removing_it_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", ENV_TOKEN)
    client = make_client(tmp_path)

    client.put("/api/settings", json={"mapillary_token": TOKEN})
    assert os.environ["MAPILLARY_ACCESS_TOKEN"] == TOKEN

    after = client.put("/api/settings", json={"clear_mapillary_token": True}).json()
    assert after["mapillary"]["source"] == "env"
    assert os.environ["MAPILLARY_ACCESS_TOKEN"] == ENV_TOKEN


def test_removing_the_only_token_unsets_it(tmp_path):
    client = make_client(tmp_path)
    client.put("/api/settings", json={"mapillary_token": TOKEN})

    after = client.put("/api/settings", json={"clear_mapillary_token": True}).json()

    assert after["mapillary"]["set"] is False
    assert "MAPILLARY_ACCESS_TOKEN" not in os.environ


@pytest.mark.parametrize(
    "token, message",
    [("   ", "Paste a token"), ("MLY|12 34", "no spaces")],
)
def test_rejects_bad_tokens(tmp_path, token, message):
    resp = make_client(tmp_path).put("/api/settings", json={"mapillary_token": token})

    assert resp.status_code == 422
    assert message in resp.text


def test_defaults_are_saved_and_validated(tmp_path):
    client = make_client(tmp_path)

    saved = client.put(
        "/api/settings", json={"defaults": {"seg_model": "tiny", "device": "cpu"}}
    ).json()
    assert saved["defaults"] == {
        "poses": "colmap",
        "remove_obstructions": True,
        "fill_obstructions": True,
        "seg_model": "tiny",
        "device": "cpu",
    }
    assert saved["mapillary"]["set"] is False  # token untouched

    bad = client.put("/api/settings", json={"defaults": {"device": "tpu"}})
    assert bad.status_code == 422
    assert (
        make_client(tmp_path).get("/api/settings").json()["defaults"]["device"] == "cpu"
    )


class FakeResponse:
    def __init__(self, status):
        self.status_code = status
        self.ok = status < 400


@pytest.mark.parametrize(
    "status, ok, message",
    [(200, True, "works"), (401, False, "rejected"), (500, False, "HTTP 500")],
)
def test_token_check(tmp_path, monkeypatch, status, ok, message):
    sent = {}

    def fake_get(url, params, timeout):
        sent.update(params)
        return FakeResponse(status)

    monkeypatch.setattr(api.settings.requests, "get", fake_get)

    resp = make_client(tmp_path).post(
        "/api/settings/mapillary/test", json={"token": TOKEN}
    )

    assert resp.json()["ok"] is ok
    assert message in resp.json()["message"]
    assert sent["access_token"] == TOKEN


def test_token_check_uses_the_current_token_and_needs_one(tmp_path, monkeypatch):
    sent = {}
    monkeypatch.setattr(
        api.settings.requests,
        "get",
        lambda url, params, timeout: sent.update(params) or FakeResponse(200),
    )
    client = make_client(tmp_path)

    missing = client.post("/api/settings/mapillary/test", json={})
    assert missing.status_code == 422

    client.put("/api/settings", json={"mapillary_token": TOKEN})
    assert client.post("/api/settings/mapillary/test", json={}).json()["ok"] is True
    assert sent["access_token"] == TOKEN


def test_jobs_use_the_token_saved_in_settings(tmp_path):
    seen = {}

    def fetch(value, state):
        seen["token"] = os.environ.get("MAPILLARY_ACCESS_TOKEN")
        return value

    def factory():
        pipeline = PipelineChain("token")
        pipeline.add_stage("fetch", fetch, label="Fetch data")
        return pipeline

    client = make_client(tmp_path, factory)
    client.put("/api/settings", json={"mapillary_token": TOKEN})

    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    for _ in range(250):
        if client.get(f"/api/jobs/{job_id}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.02)

    assert seen["token"] == TOKEN


@pytest.mark.parametrize(
    "app_env, name", [(None, ".env"), ("development", ".env.development")]
)
def test_saving_a_token_creates_the_missing_env_file(
    tmp_path, monkeypatch, app_env, name
):
    (tmp_path / ".env").unlink()
    if app_env:
        monkeypatch.setenv("APP_ENV", app_env)
    client = make_client(tmp_path)
    assert client.get("/api/settings").json()["mapillary"]["env_file_exists"] is False

    resp = client.put("/api/settings", json={"mapillary_token": TOKEN}).json()

    env_file = tmp_path / name
    assert resp["token_saved_to"] == "env_file"
    assert env_file.read_text().splitlines()[-1] == f"MAPILLARY_ACCESS_TOKEN={TOKEN}"
    if sys.platform != "win32":
        assert env_file.stat().st_mode & 0o777 == 0o600
    # It's now the env file's token, not one saved in settings.json.
    assert resp["mapillary"]["source"] == "env"
    assert resp["mapillary"]["env_file"] == name
    assert resp["mapillary"]["env_file_exists"] is True
    assert "mapillary_token" not in json.loads((tmp_path / "settings.json").read_text())
    assert os.environ["MAPILLARY_ACCESS_TOKEN"] == TOKEN

    # A fresh start reads it back from the file.
    from common.env import load_app_env

    monkeypatch.delenv("MAPILLARY_ACCESS_TOKEN")
    monkeypatch.setattr(api.app, "load_app_env", load_app_env)
    restarted = make_client(tmp_path).get("/api/settings").json()["mapillary"]
    assert restarted["source"] == "env"
    assert restarted["hint"] == "…abcd"


def test_an_existing_env_file_is_left_alone(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n")

    resp = make_client(tmp_path).put("/api/settings", json={"mapillary_token": TOKEN})

    assert resp.json()["token_saved_to"] == "settings"
    assert env_file.read_text() == "OTHER=1\n"


def test_token_goes_to_settings_when_the_env_file_cannot_be_created(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PSM_ENV_DIR", str(tmp_path / "missing-folder"))

    resp = make_client(tmp_path).put("/api/settings", json={"mapillary_token": TOKEN})

    assert resp.json()["token_saved_to"] == "settings"
    assert resp.json()["mapillary"]["source"] == "saved"
    assert os.environ["MAPILLARY_ACCESS_TOKEN"] == TOKEN
