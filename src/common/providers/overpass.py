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
        urls: tuple[str, ...] = (OVERPASS_URL, *OVERPASS_FALLBACK_URLS),
        timeout: float = 25.0,
        # one try per server: callers then fall back to the main OSM API (osm_api.py)
        rounds: int = 1,
        backoff_s: float = 10.0,
    ):
        self.urls = urls
        self.timeout = timeout
        self.rounds = rounds
        self.backoff_s = backoff_s

    def fetch(self, query: str) -> dict[str, Any]:
        """The JSON answer to an Overpass QL query, trying each server in turn, for `rounds`
        rounds with a growing pause between them (the public servers are often overloaded
        for a minute or two)."""
        if time.time() < OverpassClient._down_until:
            raise OverpassUnavailable("Overpass is down (skipping it for now)")
        errors: list[str] = []
        for k in range(self.rounds):
            if k:
                print(f"  Overpass unavailable, retrying in {self.backoff_s * k:.0f} s")
                time.sleep(self.backoff_s * k)
            result = self._round(query, errors)
            if result is not None:
                return result
        OverpassClient._down_until = time.time() + DOWN_FOR_S
        raise OverpassUnavailable(
            "All Overpass servers failed; try again later: "
            + "; ".join(errors[-len(self.urls) :])
        )

    def _round(self, query: str, errors: list[str]) -> dict[str, Any] | None:
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
        return None
