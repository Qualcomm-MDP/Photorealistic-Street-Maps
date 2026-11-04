"""Overpass QL queries against the public Overpass servers."""

import time
from typing import Any

import requests

from ..constants import OVERPASS_FALLBACK_URLS, OVERPASS_URL, USER_AGENT
from ..redact import redact

# Status codes Overpass returns when it is overloaded or rate limiting.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# After every server has failed, queries fail at once for this long (s): callers fall
# back or go without, so an outage costs one round of attempts rather than one per query.
DOWN_FOR_S = 600.0


class OverpassUnavailable(RuntimeError):
    """Every Overpass server failed (or they did within the last DOWN_FOR_S)."""


class OverpassClient:
    _down_until = 0.0  # shared by every client in the process

    def __init__(
        self,
        urls: tuple[str, ...] = (
            OVERPASS_URL,
            *OVERPASS_FALLBACK_URLS,
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ),
        timeout: float = 45.0,
    ):
        self.urls = urls
        self.timeout = timeout

    def fetch(self, query: str) -> dict[str, Any]:
        """The JSON answer to an Overpass QL query, trying each server once."""
        if time.time() < OverpassClient._down_until:
            raise OverpassUnavailable("Overpass is down (skipping it for now)")
        errors = []
        for url in self.urls:
            try:
                r = requests.post(
                    url,
                    data={"data": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout,
                )
                if r.status_code not in RETRYABLE_STATUS:
                    r.raise_for_status()
                    return r.json()
                errors.append(f"{url}: HTTP {r.status_code}")
            except (requests.RequestException, ValueError) as e:
                errors.append(f"{url}: {redact(str(e))[:80]}")
            print(f"  Overpass failed ({errors[-1]}), trying next server")
        OverpassClient._down_until = time.time() + DOWN_FOR_S
        raise OverpassUnavailable(
            "All Overpass servers failed; try again later: " + "; ".join(errors)
        )
