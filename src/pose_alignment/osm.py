"""OpenStreetMap buildings for the alignment stage: footprints (Overpass, or the main OSM API
when every Overpass server is down), their tagged heights, a raster of the distance to the
nearest wall, and which side of the camera path points lie on."""

import json
from pathlib import Path

import cv2
import numpy as np

from common.MeshUtils import building_heights  # noqa: F401  (align, plot_fit import it from here)
from common.providers.osm_api import OsmApiClient
from common.providers.overpass import OverpassClient, OverpassUnavailable


def buildings_from_elements(elements):
    """[(tags, [(lon, lat)])] from Overpass "out geom" elements: a way's outline, a
    relation's outer rings (the main pipeline's fetch stage gets the same elements)."""
    buildings = []
    for el in elements:
        if el.get("type") == "way" and "geometry" in el:
            rings = [el["geometry"]]
        elif el.get("type") == "relation":
            rings = [
                m["geometry"]
                for m in el.get("members", [])
                if m.get("role") == "outer" and "geometry" in m
            ]
        else:
            continue
        buildings += [
            (el.get("tags", {}), [(p["lon"], p["lat"]) for p in ring])
            for ring in rings
            if len(ring) >= 3
        ]
    return buildings


def fetch_buildings(bbox, cache=None):
    """OSM building footprints in bbox (min_lon, min_lat, max_lon, max_lat): [(tags, [(lon, lat)])].

    From Overpass; when every Overpass server is down, from the main OpenStreetMap API
    instead. With `cache` (a JSON path) the result is saved there and reused on reruns."""
    if cache and Path(cache).exists():
        return [
            (tags, [tuple(p) for p in ring])
            for tags, ring in json.loads(Path(cache).read_text())
        ]
    w, s, e, n = bbox
    try:
        data = OverpassClient().fetch(
            f'[out:json][timeout:90];(way["building"]({s},{w},{n},{e});'
            f'relation["building"]({s},{w},{n},{e}););out geom;'
        )
        buildings = buildings_from_elements(data["elements"])
    except OverpassUnavailable:
        print(
            "  Overpass is unavailable; fetching the buildings from the OpenStreetMap API instead"
        )
        buildings = buildings_from_elements(
            OsmApiClient().elements(bbox, lambda tags: "building" in tags)["elements"]
        )
    if cache:
        Path(cache).write_text(json.dumps(buildings))
    return buildings


class WallMap:
    """Raster distance (m) to the nearest target wall, plus an inside-building mask."""

    def __init__(self, A, B, footprints, lo, hi, res=0.25):
        self.lo, self.res = np.asarray(lo, float), res
        w, h = (np.ceil((np.asarray(hi) - lo) / res)).astype(int) + 1
        img = np.full((h, w), 255, np.uint8)
        for a, b in zip(A, B):
            cv2.line(
                img,
                tuple(np.round((a - lo) / res).astype(int)),
                tuple(np.round((b - lo) / res).astype(int)),
                0,
                1,
            )
        self.dist = cv2.distanceTransform(img, cv2.DIST_L2, 5) * res
        self.solid = np.zeros((h, w), np.uint8)
        for xy, _, _ in footprints:
            cv2.fillPoly(self.solid, [np.round((xy - lo) / res).astype(np.int32)], 1)
        # Shrink by 1 m so cameras right at a facade (sidewalk) are not penalised.
        self.solid = cv2.erode(self.solid, np.ones((9, 9), np.uint8))

    def _lookup(self, grid, xy, outside):
        ij = np.round((xy - self.lo) / self.res).astype(int)
        h, w = grid.shape
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < w) & (ij[..., 1] >= 0) & (ij[..., 1] < h)
        out = np.full(xy.shape[:-1], outside, float)
        out[ok] = grid[ij[..., 1][ok], ij[..., 0][ok]]
        return out

    def distance(self, xy):
        return self._lookup(self.dist, xy, 1e3)

    def inside(self, xy):
        return self._lookup(self.solid, xy, 0) > 0


def side_of_path(points, paths):
    """True where a point lies right of the local direction of travel at its nearest camera.
    paths: one (N, 2) array of camera positions in capture order per sequence."""
    cams = np.vstack(paths)
    tangents = np.vstack(
        [np.gradient(p, axis=0) if len(p) > 1 else np.zeros_like(p) for p in paths]
    )
    k = np.argmin(np.linalg.norm(points[:, None] - cams[None], axis=2), axis=1)
    rel, t = points - cams[k], tangents[k]
    return t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0] < 0
