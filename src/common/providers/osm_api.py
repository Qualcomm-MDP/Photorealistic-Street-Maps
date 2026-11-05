"""OSM data from the main OpenStreetMap API, the fallback when Overpass is down.

The API's map call returns everything in a box (XML, at most 50,000 nodes per request), so
the box is fetched in tiles of about 400 m, split further when a tile is refused as too
big, and the ways and relations wanted are put together with their node coordinates. The
result has the shape of an Overpass "out geom" answer ({"elements": [...]}: ways with
"geometry", relations with "members" carrying theirs), so callers parse either the same.
The API is meant for editing, not as a data feed: use it only when Overpass fails.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import numpy as np
import requests

from ..constants import USER_AGENT

OSM_API_MAP = "https://api.openstreetmap.org/api/0.6/map"
TILE_DEG = 0.004  # about 400 m
MIN_TILE_DEG = 0.0005


def _tags(el) -> dict[str, str]:
    return {t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")}


class OsmApiClient:
    def __init__(self, timeout: float = 60.0, retries: int = 3):
        self.timeout = timeout
        self.retries = retries

    def _map(self, w: float, s: float, e: float, n: float) -> list[ET.Element]:
        """The XML roots of the map call for a box (split into quarters when the API
        refuses it as too big)."""
        url = f"{OSM_API_MAP}?bbox={w:.7f},{s:.7f},{e:.7f},{n:.7f}"
        for attempt in range(self.retries):
            try:
                r = requests.get(
                    url, headers={"User-Agent": USER_AGENT}, timeout=self.timeout
                )
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
                continue
            if r.status_code == 400 and (e - w) > MIN_TILE_DEG:  # too many nodes
                mx, my = (w + e) / 2, (s + n) / 2
                return [
                    root
                    for box in (
                        (w, s, mx, my),
                        (mx, s, e, my),
                        (w, my, mx, n),
                        (mx, my, e, n),
                    )
                    for root in self._map(*box)
                ]
            if (
                r.status_code in (429, 500, 502, 503, 504)
                and attempt < self.retries - 1
            ):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return [ET.fromstring(r.content)]
        raise RuntimeError(f"OpenStreetMap API: no answer for {url}")

    def elements(
        self, bbox: tuple[float, float, float, float], keep: Callable[[dict], bool]
    ) -> dict[str, Any]:
        """{"elements": ways and relations in bbox (min_lon, min_lat, max_lon, max_lat)
        whose tags pass keep}, in Overpass "out geom" form."""
        w0, s0, e0, n0 = bbox
        nodes: dict[str, dict] = {}
        ways: dict[str, tuple[dict, list[str]]] = {}
        relations: dict[str, tuple[dict, list[tuple[str, str]]]] = {}
        for w in np.arange(w0, e0, TILE_DEG):
            for s in np.arange(s0, n0, TILE_DEG):
                for root in self._map(
                    w, s, min(w + TILE_DEG, e0), min(s + TILE_DEG, n0)
                ):
                    for nd in root.iter("node"):
                        nodes[nd.attrib["id"]] = {
                            "lat": float(nd.attrib["lat"]),
                            "lon": float(nd.attrib["lon"]),
                        }
                    for way in root.iter("way"):
                        ways[way.attrib["id"]] = (
                            _tags(way),
                            [nd.attrib["ref"] for nd in way.findall("nd")],
                        )
                    for rel in root.iter("relation"):
                        relations[rel.attrib["id"]] = (
                            _tags(rel),
                            [
                                (m.attrib["ref"], m.attrib["role"])
                                for m in rel.findall("member")
                                if m.attrib["type"] == "way"
                            ],
                        )

        def geometry(refs):
            return [nodes[r] for r in refs] if all(r in nodes for r in refs) else None

        out = []
        for wid, (tags, refs) in ways.items():
            if keep(tags) and (geom := geometry(refs)):
                out.append(
                    {"type": "way", "id": int(wid), "tags": tags, "geometry": geom}
                )
        for rid, (tags, members) in relations.items():
            if not keep(tags):
                continue
            mem = [
                {"type": "way", "ref": int(ref), "role": role, "geometry": geom}
                for ref, role in members
                if ref in ways and (geom := geometry(ways[ref][1]))
            ]
            if mem:
                out.append(
                    {"type": "relation", "id": int(rid), "tags": tags, "members": mem}
                )
        return {"elements": out}
