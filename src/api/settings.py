"""Settings people change in the web UI, saved as settings.json next to the job data.

The Mapillary token can come from two places: the environment (the APP_ENV's
env file, e.g. .env, or `-e`), or the web UI. A token saved in the UI wins;
removing it falls back to the environment's again. Either way it ends up in
MAPILLARY_ACCESS_TOKEN, which is where the pipeline (and the log redaction)
read it from.

When the env file doesn't exist yet, a token saved in the UI creates it
instead, so it lands where people expect it. If that file can't be written,
the token goes into settings.json.

The settings response only says whether a token is set, where it came from
and its last few characters. The token itself is sent only by its own
endpoint, which the Settings dialog calls to show it (masked until "Show").
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import requests

from common.constants import MAPILLARY_URL
from common.env import app_env_file, app_env_path
from common.redact import redact

TOKEN_ENV = "MAPILLARY_ACCESS_TOKEN"
DEFAULTS: dict[str, Any] = {
    "remove_obstructions": True,
    "seg_model": "large",
    "device": "auto",
}
# A small area that always has imagery, so a valid token gets a 200 quickly.
_TEST_BBOX = "-83.7170,42.2918,-83.7160,42.2924"


class Settings:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        # The token from .env / the environment, before any saved one replaces it.
        self._env_token = os.environ.get(TOKEN_ENV, "").strip() or None
        self._data = self._read()
        self._apply_token()

    # ---- reading -----------------------------------------------------------

    def public(self) -> dict[str, Any]:
        """The settings, without the token itself."""
        with self._lock:
            saved = self._data.get("mapillary_token")
            token = saved or self._env_token
            return {
                "mapillary": {
                    "set": bool(token),
                    "source": "saved" if saved else "env" if token else None,
                    "hint": _hint(token),
                    "env_available": bool(self._env_token),
                    # Where an environment token comes from, e.g. .env.development.
                    "env_file": app_env_file(),
                    "env_file_exists": app_env_path().is_file(),
                },
                "defaults": {**DEFAULTS, **self._data.get("defaults", {})},
            }

    def token(self) -> str | None:
        with self._lock:
            return self._data.get("mapillary_token") or self._env_token

    # ---- changing ----------------------------------------------------------

    def update(
        self,
        mapillary_token: str | None = None,
        clear_token: bool = False,
        defaults: dict[str, Any] | None = None,
    ) -> str | None:
        """Apply the changes. For a new token, returns where it was saved:
        "env_file" (a newly created env file) or "settings" (settings.json)."""
        saved_to = None
        with self._lock:
            if clear_token:
                self._data.pop("mapillary_token", None)
            elif mapillary_token is not None:
                saved_to = self._save_token(mapillary_token.strip())
            if defaults is not None:
                self._data["defaults"] = {
                    **self._data.get("defaults", {}),
                    **{k: v for k, v in defaults.items() if k in DEFAULTS},
                }
            self._write()
            self._apply_token()
        return saved_to

    # ---- internals ---------------------------------------------------------

    def _save_token(self, token: str) -> str:
        env_path = app_env_path()
        if not env_path.exists():
            try:
                _create_env_file(env_path, token)
            except OSError:
                pass  # e.g. a read-only folder: keep it in settings.json
            else:
                # It's now the environment's token, as if it had always been
                # in the file; nothing to override it with.
                self._env_token = token
                self._data.pop("mapillary_token", None)
                return "env_file"
        self._data["mapillary_token"] = token
        return "settings"

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        # Created readable by this user only: it can hold the token.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self.path)

    def _apply_token(self) -> None:
        token = self._data.get("mapillary_token") or self._env_token
        if token:
            os.environ[TOKEN_ENV] = token
        else:
            os.environ.pop(TOKEN_ENV, None)


def _create_env_file(path: Path, token: str) -> None:
    # O_EXCL: never overwrite a file that appeared meanwhile. Readable by this
    # user only, since it holds the token.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"# Created by PSM's Settings.\n{TOKEN_ENV}={token}\n")


def _hint(token: str | None) -> str | None:
    return f"…{token[-4:]}" if token and len(token) >= 8 else None


def check_token(token: str) -> tuple[bool, str]:
    """Ask Mapillary whether the token works: (ok, message for the user)."""
    params: dict[str, Any] = {
        "access_token": token,
        "bbox": _TEST_BBOX,
        "fields": "id",
        "limit": 1,
    }
    try:
        resp = requests.get(MAPILLARY_URL, params=params, timeout=10)
    except requests.RequestException as exc:
        return False, redact(f"Couldn't reach Mapillary: {exc}")
    if resp.ok:
        return True, "The token works."
    if resp.status_code in (400, 401, 403):
        return False, "Mapillary rejected this token. Check that you copied all of it."
    return (
        False,
        f"Mapillary answered with an error (HTTP {resp.status_code}). Try again later.",
    )
