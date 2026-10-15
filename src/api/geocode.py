"""Address search on OpenStreetMap data.

search(): full search through Nominatim. The public server allows at most one
request per second, asks clients to cache results, and forbids autocomplete:
https://operations.osmfoundation.org/policies/nominatim/
so it's only used when the user submits a search.

suggest(): suggestions while typing, through Photon (komoot), which is built
for search-as-you-type on OSM data: https://photon.komoot.io
The public Photon server is for light, fair use; self-host it for heavy traffic.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

import requests

from common.constants import USER_AGENT

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"
_MIN_INTERVAL_S = 1.0
_CACHE_SIZE = 256

_lock = threading.Lock()
_last_request = 0.0
_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_suggest_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()


def _remember(cache: OrderedDict, key: str, value: list[dict[str, Any]]) -> None:
    cache[key] = value
    if len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)


def search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    key = f"{limit}:{query.strip().lower()}"
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

        global _last_request
        wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={"q": query, "format": "jsonv2", "limit": limit},
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
        finally:
            _last_request = time.monotonic()
        resp.raise_for_status()

        results = [_to_place(r) for r in resp.json()]
        _remember(_cache, key, results)
        return results


def suggest(
    query: str, lat: float | None = None, lon: float | None = None, limit: int = 6
) -> list[dict[str, Any]]:
    """Places matching a partial query, ranked toward (lat, lon) when given."""
    # Round the bias point so small map pans still hit the cache.
    bias = (
        (round(lat, 2), round(lon, 2)) if lat is not None and lon is not None else None
    )
    key = f"{limit}:{bias}:{query.strip().lower()}"
    if key in _suggest_cache:
        _suggest_cache.move_to_end(key)
        return _suggest_cache[key]

    params: dict[str, Any] = {"q": query, "limit": limit, "lang": "en"}
    if bias:
        params["lat"], params["lon"] = bias
    resp = requests.get(
        PHOTON_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=8
    )
    resp.raise_for_status()
    results = [_photon_to_place(f) for f in resp.json().get("features", [])]
    _remember(_suggest_cache, key, results)
    return results


def _to_place(r: dict[str, Any]) -> dict[str, Any]:
    # Nominatim's boundingbox is [south, north, west, east] as strings.
    south, north, west, east = (float(v) for v in r["boundingbox"])
    display = r.get("display_name", "")
    return {
        "name": r.get("name") or display.split(",")[0],
        "address": display,
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "bbox": {"west": west, "south": south, "east": east, "north": north},
    }


def _photon_to_place(feature: dict[str, Any]) -> dict[str, Any]:
    p = feature.get("properties", {})
    lon, lat = feature["geometry"]["coordinates"]
    street = " ".join(x for x in (p.get("housenumber"), p.get("street")) if x)
    name = p.get("name") or street or p.get("city") or "Unnamed place"
    parts = [
        street if street != name else None,
        p.get("city"),
        p.get("state"),
        p.get("country"),
    ]
    address = ", ".join([name] + [x for x in parts if x and x != name])
    # Photon's extent is [lon, lat, lon, lat] for two opposite corners.
    extent = p.get("extent")
    if extent and len(extent) == 4:
        west, east = sorted((extent[0], extent[2]))
        south, north = sorted((extent[1], extent[3]))
    else:
        west = east = lon
        south = north = lat
    return {
        "name": name,
        "address": address,
        "lat": float(lat),
        "lon": float(lon),
        "bbox": {"west": west, "south": south, "east": east, "north": north},
    }
