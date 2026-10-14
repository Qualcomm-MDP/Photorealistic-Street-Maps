"""Stage 5 (OSM alignment): place the COLMAP reconstructions on OpenStreetMap buildings and
write the corrected camera poses.

OSMAlignmentLayer.align() aligns one leg: the COLMAP models of one rig's cameras.
  1. _level_up_vector         turn each model upright (walls vertical, road level)
  2. _tie_rig_cameras         map the rig's other cameras onto the first by capture time
  3. _mapillary_only          the "before": the camera path fitted to Mapillary's positions,
                              which also gives the starting scale (metres per model unit)
  4. _extract_facades_ransac  building points in plan view -> facade line segments
  5. _get_nearby_osm_walls    street-facing walls of the OSM footprints near the track
  6. _fit_2d_similarity       heading, scale and position putting the facades on the walls
  7. _format_output           transforms, statistics and plan-view data for the images

align_roi() runs every leg of an ROI (legs.json from the COLMAP layer) and writes:
  data/alignment/<sequence folder>.json  corrected cameras of one sequence: per frame the
                                         pipeline pose (lat/lon, height above the road, ENU,
                                         camera-to-ENU rotation, Blender matrix,
                                         heading/pitch/roll, intrinsics), raw GPS and
                                         Mapillary's position for comparison, a reliable
                                         flag, and why a frame was not aligned
  data/alignment/summary.json            per-leg results and flags
  data/alignment/legs/<leg>.json         per-leg statistics and transforms
  results/alignment_overview.png         before/after map of the whole ROI
  results/raw_gps_vs_pipeline.png        raw GPS vs pipeline positions and headings
  results/legs/<leg>_alignment.png       before/after map of one leg

Usage:
  python osm_alignment_layer.py rois/mapillary_roi_1
  python osm_alignment_layer.py rois/mapillary_roi_1 --camera-height 1.4
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from matplotlib.path import Path as PolyPath
from scipy.optimize import least_squares

from alignment_plots import plot_before_after, plot_gps_vs_pipeline
from common import (MASK_BUILDING, MASK_ROAD, LocalFrame, RoiPaths, apply_2d, fit_plane, gravity_frame, label_points,
                    load_cameras, load_points, overpass, planarity, ransac_lines, read_camera, segment_distances,
                    similarity_2d, split_segments, wrap_deg)

DEFAULTS = {
    "level_height": 3.5,       # metres per building level
    "default_levels": 3,       # levels for buildings without height tags
    "margin": 60.0,            # OSM search margin around the track (m)
    "max_grade": 0.08,         # steepest plausible grade along the path
    "class_fraction": 0.5,     # min share of a point's observations on building pixels
    # Facade extraction (m).
    "min_track": 3,            # min frames observing a facade point
    "min_height": 2.5,         # facade points above cars
    "max_height": 40.0,
    "facade_thresh": 0.6,      # RANSAC inlier distance
    "facade_min_length": 4.0,
    "facade_min_points": 25,
    "facade_gap": 4.0,         # split a facade at gaps longer than this
    "min_planarity": 0.5,      # reject segments that are not thin (tree canopies)
    # Fit.
    "wall_scale": 0.75,        # Cauchy scale for facade-to-wall distance (m)
    "match_angle": 20.0,       # max angle between a facade and its wall (deg)
    "match_dist": 6.0,         # max distance to assign a facade sample to a wall (m)
    "search_deg": 15.0,        # heading search range (deg)
    "search_m": 20.0,          # position search range (m)
    "shift_prior": 15.0,       # search: a shift this far costs 0.5 m of mean wall distance
    "gps_sigma": 25.0,         # uncertainty (m) of the Mapillary mean position
    "scale_range": None,       # scale search range vs Mapillary's (default: by facade sides)
    "scale_sigma": None,       # relative scale uncertainty (default: by facade sides)
}
AXIS_FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV camera (+z forward, +y down) -> Blender (-z forward, +y up)


class AlignmentError(ValueError):
    """A leg that cannot be aligned; `label` names the camera at fault, if one is."""

    def __init__(self, message, label=None):
        super().__init__(message)
        self.label = label


# ----------------------------------------------------------------------------- OSM

def fetch_buildings(bbox):
    """OSM building footprints in bbox (min_lon, min_lat, max_lon, max_lat): [(tags, [(lon, lat)])]."""
    w, s, e, n = bbox
    data = overpass(f'[out:json][timeout:90];(way["building"]({s},{w},{n},{e});'
                    f'relation["building"]({s},{w},{n},{e}););out geom;')
    buildings = []
    for el in data["elements"]:
        rings = [el["geometry"]] if el["type"] == "way" else [
            m["geometry"] for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
        buildings += [(el.get("tags", {}), [(p["lon"], p["lat"]) for p in ring]) for ring in rings]
    return buildings


def building_heights(tags, level_height, default_levels):
    """(base, top, source) in metres above ground from OSM tags."""
    def metres(value):
        try:
            return float(str(value).lower().replace("m", "").split(";")[0].strip())
        except ValueError:
            return None

    base = metres(tags.get("min_height"))
    if base is None and tags.get("building:min_level"):
        base = (metres(tags["building:min_level"]) or 0) * level_height
    top, source = metres(tags.get("height") or tags.get("building:height")), "height"
    if top is None and tags.get("building:levels") and metres(tags["building:levels"]):
        top, source = metres(tags["building:levels"]) * level_height, "levels"
    if top is None:
        top, source = default_levels * level_height, "default"
    return base or 0.0, top, source


def visible_walls(A, B, origins, n_rays=720, max_range=80.0):
    """How many rays from the origins hit each wall segment A->B first."""
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    e = B - A
    denom = rays[:, None, 0] * e[None, :, 1] - rays[:, None, 1] * e[None, :, 0]
    hits = np.zeros(len(A), int)
    with np.errstate(divide="ignore", invalid="ignore"):
        for o in origins:
            w = A - o
            t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0])[None] / denom  # distance along the ray
            u = (w[None, :, 0] * rays[:, None, 1] - w[None, :, 1] * rays[:, None, 0]) / denom  # position on the wall
            t = np.where((t > 1e-6) & (t < max_range) & (u >= 0) & (u <= 1), t, np.inf)
            first = np.argmin(t, axis=1)
            np.add.at(hits, first[np.isfinite(t[np.arange(len(rays)), first])], 1)
    return hits


class WallMap:
    """Raster distance (m) to the nearest target wall, plus an inside-building mask."""

    def __init__(self, A, B, footprints, lo, hi, res=0.25):
        self.lo, self.res = np.asarray(lo, float), res
        w, h = (np.ceil((np.asarray(hi) - lo) / res)).astype(int) + 1
        img = np.full((h, w), 255, np.uint8)
        for a, b in zip(A, B):
            cv2.line(img, tuple(np.round((a - lo) / res).astype(int)), tuple(np.round((b - lo) / res).astype(int)), 0, 1)
        self.dist = cv2.distanceTransform(img, cv2.DIST_L2, 5) * res
        solid = np.zeros((h, w), np.uint8)
        for xy, _, _ in footprints:
            cv2.fillPoly(solid, [np.round((xy - lo) / res).astype(np.int32)], 1)
        self.solid = cv2.erode(solid, np.ones((9, 9), np.uint8))  # 1 m in, so sidewalk cameras are not "inside"

    def _lookup(self, grid, xy, outside):
        ij = np.round((xy - self.lo) / self.res).astype(int)
        ok = (ij[..., 0] >= 0) & (ij[..., 0] < grid.shape[1]) & (ij[..., 1] >= 0) & (ij[..., 1] < grid.shape[0])
        out = np.full(xy.shape[:-1], outside, float)
        out[ok] = grid[ij[..., 1][ok], ij[..., 0][ok]]
        return out

    def distance(self, xy):
        return self._lookup(self.dist, xy, 1e3)

    def inside(self, xy):
        return self._lookup(self.solid, xy, 0) > 0


# ----------------------------------------------------------------------------- one leg

def _apply_3d(p, pts):
    """A 2D similarity on x, y, with z scaled by the same factor."""
    return np.column_stack([apply_2d(p, pts[:, :2]), math.exp(p[1]) * pts[:, 2]])


def _rotate_dirs(p, d):
    c, s = math.cos(p[0]), math.sin(p[0])
    return d @ np.array([[c, s], [-s, c]])


class OSMAlignmentLayer:
    """Aligns the COLMAP models of one leg to OSM building walls (see the module docstring)."""

    def __init__(self, camera_height=1.4, **config):
        self.camera_height = camera_height
        self.cfg = SimpleNamespace(**{**DEFAULTS, **config})

    def align(self, inputs, buildings=None):
        """inputs: [{"label", "model" (COLMAP TXT folder), "frames" (cameras.json frames by
        file), "masks" (segmentation folder)}], the rig's reference camera first.
        buildings: fetch_buildings() output around the track (fetched if None).
        Returns {"transforms", "stats", "plan", "enu_origin_lon_lat", "camera_height_m"};
        raises AlignmentError if the leg cannot be aligned."""
        models = [self._level_up_vector(inp) for inp in inputs]
        ties = self._tie_rig_cameras(models)
        start = self._mapillary_only(models)
        facades = self._extract_facades_ransac(models, start)
        walls = self._get_nearby_osm_walls(buildings, start)
        params = self._fit_2d_similarity(facades, walls, start)
        return self._format_output(models, ties, start, facades, walls, params)

    # 1. ---------------------------------------------------------------------------------
    def _level_up_vector(self, inp):
        """Load a model and rotate it to (right, forward, up), leaving heading, scale and 2D
        position as the unknowns. Building walls are vertical, so up is perpendicular to
        the normals of the large near-vertical planes among building points; of those
        directions, the one closest to the road plane's normal is taken (a road normal alone
        tilts with the street's slope; one wall direction leaves a rotation free). Walls
        along a street fix the cross-street tilt only; if the along-street tilt then implies
        a grade steeper than max_grade, it is removed (path level along its length).
        Without enough road points, up comes from the travel direction and camera axes."""
        model = Path(inp["model"])
        names, centres, rots = load_cameras(model)
        xyz, rgb, track = load_points(model)
        facade = label_points(model, inp["masks"], MASK_BUILDING, self.cfg.class_fraction)[0]
        road = label_points(model, inp["masks"], MASK_ROAD)[0]
        origin, basis, level = self._upright_frame(centres, rots, xyz[road], xyz[facade & (track >= 3)])
        meta = inp["frames"]
        return SimpleNamespace(
            label=inp["label"], model=model, origin=origin, basis=basis, level=level, rel=np.zeros(4),
            pts=(xyz - origin) @ basis.T, cams=(centres - origin) @ basis.T, rgb=rgb, track=track, facade=facade,
            times=np.array([meta[n]["captured_at_ms"] / 1000 for n in names]),
            mapillary=[(meta[n]["lon"], meta[n]["lat"]) for n in names],
            raw_gps=[tuple(meta[n]["gps"]) for n in names if meta[n].get("gps")])

    def _upright_frame(self, centres, rots, road_pts, building_pts, min_points=50):
        origin, basis = gravity_frame(centres, rots)  # fallback: travel direction and camera axes
        span = max(np.linalg.norm(centres[-1] - centres[0]), 1e-9)
        rng, thresh = np.random.default_rng(0), 0.004 * span

        def near(P, radius):
            return P[np.min(np.linalg.norm(P[:, None] - centres[None, ::2], axis=2), axis=1) < radius * span] if len(P) else P

        P = near(road_pts, 0.3)
        road_up, inl = fit_plane(P, thresh, rng) if len(P) >= min_points else (None, None)
        if road_up is None or inl.sum() < min_points:
            return origin, basis, {"level": "travel direction", "road_points": int(len(P))}
        road_up = road_up if road_up @ basis[2] > 0 else -road_up

        walls, W = [], near(building_pts, 0.6)  # near-vertical planes among building points (6 tries)
        for _ in range(6):
            if len(W) < min_points:
                break
            n, inl_w = fit_plane(W, thresh, rng, iters=1500)
            if n is None or inl_w.sum() < min_points:
                break
            if abs(n @ road_up) < math.sin(math.radians(25)):
                walls.append((n, int(inl_w.sum())))
            W = W[~inl_w]

        def up_from(walls):
            # Unit u minimising sum w_i (n_i . u)^2 - 0.05 (road_up . u)^2: among the
            # directions the walls allow, the one nearest the road normal.
            total = sum(k for _, k in walls)
            M = sum((k / total) * np.outer(n, n) for n, k in walls) - 0.05 * np.outer(road_up, road_up)
            u = np.linalg.eigh(M)[1][:, 0]
            return u if u @ road_up > 0 else -u

        up = road_up
        if walls:
            up = up_from(walls)
            kept = [(n, k) for n, k in walls if abs(n @ up) < math.sin(math.radians(6))]  # drop awnings, signs
            if kept and len(kept) < len(walls):
                walls, up = kept, up_from(kept)
        travel = (centres[-1] - centres[0]) / span
        grade = float(np.tan(np.arcsin(np.clip(travel @ up, -1, 1))))
        capped = abs(grade) > self.cfg.max_grade
        if capped:
            up = up - (up @ travel) * travel  # keep the cross-street tilt, level along the path
            up /= np.linalg.norm(up)
        forward = (centres[-1] - centres[0]) - ((centres[-1] - centres[0]) @ up) * up
        forward /= np.linalg.norm(forward)
        return origin, np.stack([np.cross(forward, up), forward, up]), {
            "level": ("walls + road plane" if walls else "road plane") + (" (grade capped)" if capped else ""),
            "road_points": int(inl.sum()), "walls": len(walls),
            "path_grade_before_cap": round(grade, 4), "grade_capped": bool(capped)}

    # 2. ---------------------------------------------------------------------------------
    def _tie_rig_cameras(self, models):
        """Map each extra camera's model onto the reference camera's levelled frame: a 2D
        similarity taking its camera path onto the reference path interpolated at its
        capture times (same vehicle, same moment, same place). Returns [(model, rms, n)]."""
        ref, ties = models[0], []
        order = np.argsort(ref.times)
        t, C = ref.times[order], ref.cams[order, :2]
        for m in models[1:]:
            overlap = (m.times >= t[0]) & (m.times <= t[-1])
            if overlap.sum() < 5:
                raise AlignmentError(f"{m.label}: fewer than 5 frames overlap {ref.label} in time", m.label)
            target = np.column_stack([np.interp(m.times[overlap], t, C[:, 0]), np.interp(m.times[overlap], t, C[:, 1])])
            s, a, tr = similarity_2d(m.cams[overlap, :2], target)
            m.rel = np.array([a, math.log(s), *tr])
            rms = float(np.sqrt(np.mean(np.sum((apply_2d(m.rel, m.cams[overlap, :2]) - target) ** 2, axis=1))))
            m.pts, m.cams = _apply_3d(m.rel, m.pts), _apply_3d(m.rel, m.cams)
            ties.append((m, rms, int(overlap.sum())))
        return ties

    # 3. ---------------------------------------------------------------------------------
    def _mapillary_only(self, models):
        """Local ENU metres (origin: the reference camera's first frame) and the 2D similarity
        mapping the levelled camera path onto Mapillary's computed positions (OpenSfM). Its
        scale converts model units to metres for the facade thresholds."""
        enu = LocalFrame(*models[0].mapillary[0])
        cams = np.vstack([m.cams for m in models])
        gps = np.array([enu.xy(*ll) for m in models for ll in m.mapillary])
        s, a, t = similarity_2d(cams[:, :2], gps)
        raw = np.array([enu.xy(*ll) for m in models for ll in m.raw_gps]).reshape(-1, 2)
        return SimpleNamespace(enu=enu, cams=cams, gps=gps, raw=raw, scale=s, params=np.array([a, math.log(s), *t]))

    # 4. ---------------------------------------------------------------------------------
    def _extract_facades_ransac(self, models, start):
        """Facade line segments in plan view, sampled about every metre. Points must be
        labelled building, seen by >= min_track frames and between min_height and max_height
        (cars and noisy two-view points out). Sequential RANSAC finds lines; lines are split
        at gaps and kept only if thin (planarity), which rejects tree canopies. Each model
        is searched on its own: lines across both sides of the street would mix."""
        c, s = self.cfg, start.scale
        segs, n_points = [], 0
        for m in models:
            height = s * m.pts[:, 2] + self.camera_height
            usable = m.facade & (m.track >= c.min_track) & (height > c.min_height) & (height < c.max_height)
            xy, thresh = m.pts[usable, :2], c.facade_thresh / s
            n_points += int(usable.sum())
            for centre, d, inl in ransac_lines(xy, thresh, c.facade_min_points):
                along = (xy[inl] - centre) @ d
                for run in split_segments(along, c.facade_gap / s, c.facade_min_points, c.facade_min_length / s):
                    if planarity(xy, centre, d, along[run].min(), along[run].max(), thresh) < c.min_planarity:
                        continue
                    P = xy[inl[run]]
                    mid = P.mean(axis=0)
                    direction = np.linalg.svd(P - mid)[2][0]
                    proj = (P - mid) @ direction
                    segs.append((mid + direction * proj.min(), mid + direction * proj.max()))
        if not segs:
            raise AlignmentError(f"no facade segments found ({n_points} usable building points)")
        samples, dirs = [], []
        for a, b in segs:
            n = max(2, int(np.linalg.norm(b - a) * s))
            samples.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
            dirs.append(np.tile((b - a) / np.linalg.norm(b - a), (n, 1)))
        samples = np.vstack(samples)

        # Side of the camera path (right of the local travel direction at the nearest camera).
        # Facades on both sides fix the scale; on one side, scale and cross-street position
        # trade off, so the scale is held to Mapillary's track.
        paths = [m.cams[:, :2] for m in models]
        cams = np.vstack(paths)
        tangents = np.vstack([np.gradient(p, axis=0) if len(p) > 1 else np.zeros_like(p) for p in paths])
        k = np.argmin(np.linalg.norm(samples[:, None] - cams[None], axis=2), axis=1)
        rel = samples - cams[k]
        right = tangents[k, 0] * rel[:, 1] - tangents[k, 1] * rel[:, 0] < 0
        both = min(right.mean(), 1 - right.mean()) >= 0.2
        print(f"    facades: {len(segs)} segments from {n_points} building points, "
              f"total {sum(np.linalg.norm(b - a) for a, b in segs) * s:.0f} m, {'both sides' if both else 'one side'}")
        return SimpleNamespace(segs=segs, samples=samples, dirs=np.vstack(dirs), right=right, both_sides=both)

    # 5. ---------------------------------------------------------------------------------
    def _get_nearby_osm_walls(self, buildings, start):
        """OSM footprints near the track and their street-facing walls: rays cast from around
        Mapillary's track (it can be metres off) mark the walls they hit first, so back
        walls, shared walls and the far side of a block are never targets."""
        c, enu = self.cfg, start.enu
        lo = start.gps.min(axis=0) - c.margin - c.search_m
        hi = start.gps.max(axis=0) + c.margin + c.search_m
        if buildings is None:
            buildings = fetch_buildings((*enu.lonlat(*lo), *enu.lonlat(*hi)))
        footprints, sources = [], {}
        for tags, ring in buildings:
            xy = np.array([enu.xy(lon, lat) for lon, lat in ring])
            if np.all(xy.max(axis=0) >= lo) and np.all(xy.min(axis=0) <= hi):
                base, top, source = building_heights(tags, c.level_height, c.default_levels)
                sources[source] = sources.get(source, 0) + 1
                footprints.append((xy, base, top))
        if not footprints:
            raise AlignmentError("no OSM buildings near the track")
        A = np.vstack([xy[:-1] for xy, _, _ in footprints])
        B = np.vstack([xy[1:] for xy, _, _ in footprints])
        d = (start.gps[-1] - start.gps[0]) / np.linalg.norm(start.gps[-1] - start.gps[0])
        origins = np.vstack([start.gps[::2] + k * np.array([-d[1], d[0]]) for k in (-8, -4, 0, 4, 8)])
        inside = np.any([PolyPath(xy).contains_points(origins) for xy, _, _ in footprints], axis=0)
        visible = (visible_walls(A, B, origins[~inside]) >= 3) & (np.linalg.norm(B - A, axis=1) >= 1.5)
        if not visible.any():
            raise AlignmentError("no street-facing OSM walls near the track")
        VA, VB = A[visible], B[visible]
        wall_dir = (VB - VA) / np.linalg.norm(VB - VA, axis=1)[:, None]
        return SimpleNamespace(footprints=footprints, sources=sources, visible=visible, VA=VA, VB=VB, dir=wall_dir,
                               normal=np.stack([-wall_dir[:, 1], wall_dir[:, 0]], axis=1),
                               map=WallMap(VA, VB, footprints, lo, hi))

    # 6. ---------------------------------------------------------------------------------
    def _fit_2d_similarity(self, facades, walls, start):
        """Heading, scale and position [angle, log scale, tx, ty] putting the facades on the
        walls: a global grid search, then robust refinement."""
        return self._refine(self._global_search(facades, walls, start), facades, walls, start)

    def _scale_freedom(self, facades):
        """(search range as factors of Mapillary's scale, relative prior sigma)."""
        both = facades.both_sides
        return (self.cfg.scale_range or ([0.7, 1.4] if both else [0.85, 1.2]),
                self.cfg.scale_sigma or (0.3 if both else 0.1))

    def _global_search(self, facades, walls, start):
        """Grid over heading (+-search_deg), scale and position (+-search_m). Score: mean
        distance of the facade samples to the visible walls (capped at 3 m), plus a penalty
        for cameras inside buildings and a mild prior on moving away from Mapillary's
        positions (it breaks ties along the street)."""
        c = self.cfg
        cam_mid, gps_mid = start.cams[:, :2].mean(axis=0), start.gps.mean(axis=0)
        offsets = np.arange(-c.search_m, c.search_m + 1e-9, 2.0)
        shift_cost = 0.5 * (np.hypot(*np.meshgrid(offsets, offsets, indexing="ij")) / c.shift_prior) ** 2
        scales = start.scale * np.exp(np.linspace(*np.log(self._scale_freedom(facades)[0]), 15))
        best = (np.inf, None)
        for dang in np.radians(np.arange(-c.search_deg, c.search_deg + 1e-9, 1.0)):
            ang = start.params[0] + dang
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            for sc in scales:
                t = gps_mid - sc * R @ cam_mid  # rotate and scale about the camera centroid
                p = [ang, math.log(sc), *t]
                samples, cams = apply_2d(p, facades.samples), apply_2d(p, start.cams[:, :2])
                for ix, dx in enumerate(offsets):
                    shift = np.stack([np.full_like(offsets, dx), offsets], axis=1)[:, None]
                    score = np.minimum(walls.map.distance(samples[None] + shift), 3.0).mean(axis=1)
                    score += 5.0 * walls.map.inside(cams[None] + shift).mean(axis=1)
                    score += shift_cost[ix]
                    j = int(np.argmin(score))
                    if score[j] < best[0]:
                        best = (score[j], np.array([ang, math.log(sc), t[0] + dx, t[1] + offsets[j]]))
        return best[1]

    def _wall_distances(self, p, facades, walls):
        """Distance from each facade sample to each wall of a compatible direction (inf otherwise)."""
        d = segment_distances(apply_2d(p, facades.samples), walls.VA, walls.VB)
        d[np.abs(_rotate_dirs(p, facades.dirs) @ walls.dir.T) <= math.cos(math.radians(self.cfg.match_angle))] = np.inf
        return d

    def _refine(self, params, facades, walls, start):
        """Iterated: each facade sample is assigned to the nearest direction-compatible wall,
        and its residual is the distance to that wall's infinite line (a partly seen facade
        can slide along its wall). Robust (Cauchy) least squares, with weak priors on
        Mapillary's mean position and scale, within +-5 deg, x1.25 and +-5 m of the search."""
        c, gps_mid = self.cfg, start.gps.mean(axis=0)
        scale_sigma = self._scale_freedom(facades)[1]
        lower = [params[0] - math.radians(5), params[1] - math.log(1.25), params[2] - 5, params[3] - 5]
        upper = [params[0] + math.radians(5), params[1] + math.log(1.25), params[2] + 5, params[3] + 5]
        for _ in range(10):
            d = self._wall_distances(params, facades, walls)
            k = np.argmin(d, axis=1)
            matched = d[np.arange(len(d)), k] < c.match_dist
            S, A, N = facades.samples[matched], walls.VA[k[matched]], walls.normal[k[matched]]
            r0 = np.abs(np.sum((apply_2d(params, S) - A) * N, axis=1)) / c.wall_scale
            w = 1.0 / np.sqrt(1.0 + r0 ** 2)  # square root of the Cauchy IRLS weights
            n = math.sqrt(max(int(matched.sum()), 1))

            def residuals(p):
                return np.r_[w * np.sum((apply_2d(p, S) - A) * N, axis=1) / c.wall_scale,
                             n * (apply_2d(p, start.cams[:, :2]).mean(axis=0) - gps_mid) / c.gps_sigma,
                             n * (p[1] - start.params[1]) / scale_sigma]

            params = least_squares(residuals, np.clip(params, lower, upper), bounds=(lower, upper),
                                   x_scale=[0.01, 0.01, 1, 1]).x
        return params

    # 7. ---------------------------------------------------------------------------------
    def _format_output(self, models, ties, start, facades, walls, params):
        """Per-model transforms (COLMAP -> ENU), statistics and plan-view data in lon/lat."""
        def facade_stats(p):
            d = np.minimum(self._wall_distances(p, facades, walls).min(axis=1), 50.0)  # no wall: 50 m
            sides = (("left", ~facades.right), ("right", facades.right), ("all", np.ones(len(d), bool)))
            return d, {side: {"samples": int(m.sum()), "median_m": float(np.median(d[m])),
                              "within_1m": float(np.mean(d[m] < 1.0)), "within_2m": float(np.mean(d[m] < 2.0))}
                       for side, m in sides if m.any()}

        p0 = start.params
        d_before, before = facade_stats(p0)
        d_after, after = facade_stats(params)
        s = math.exp(params[1])
        stats = {
            "cameras": [m.label for m in models],
            "time_ties": [{"label": m.label, "frames": n, "path_misfit_rms_m": rms * start.scale,
                           "relative_scale": math.exp(m.rel[1]), "relative_heading_deg": math.degrees(m.rel[0])}
                          for m, rms, n in ties],
            "osm_buildings": len(walls.footprints), "osm_height_sources": walls.sources,
            "street_facing_walls": int(walls.visible.sum()), "facade_segments": len(facades.segs),
            "facades_both_sides": bool(facades.both_sides),
            "facade_to_wall_mapillary_only": before, "facade_to_wall_aligned": after,
            "levelling": [m.level for m in models],
            "scale_m_per_unit": s, "scale_vs_mapillary": math.exp(params[1] - p0[1]),
            "heading_change_deg": math.degrees(params[0] - p0[0]), "shift_vs_mapillary_m": (params[2:] - p0[2:]).tolist(),
            "cameras_inside_buildings": int(walls.map.inside(apply_2d(params, start.cams[:, :2])).sum()),
            "camera_height_m": self.camera_height,
        }
        for m, rms, _ in ties:
            print(f"    tied {m.label} to {models[0].label} by capture time: path misfit RMS {rms * start.scale:.2f} m")
        print(f"    facade-to-wall median: Mapillary-only {before['all']['median_m']:.2f} m -> aligned "
              f"{after['all']['median_m']:.2f} m ({100 * after['all']['within_1m']:.0f}% within 1 m); "
              f"scale x{stats['scale_vs_mapillary']:.2f}, heading {stats['heading_change_deg']:+.1f} deg")

        # Composed per model: the time tie into the reference frame, then the OSM alignment.
        transforms = [{"label": m.label, "model": str(m.model), "colmap_origin": m.origin.tolist(),
                       "basis": m.basis.tolist(), "scale_m_per_unit": s * math.exp(m.rel[1]),
                       "angle_rad": float(params[0] + m.rel[0]), "translation_m": apply_2d(params, m.rel[None, 2:])[0].tolist()}
                      for m in models]

        # Plan-view data in lon/lat for the images (several legs share one map).
        def lonlat(xy):
            return [[round(v, 8) for v in start.enu.lonlat(x, y)] for x, y in np.asarray(xy, float).reshape(-1, 2)]

        def segments(p):
            return [lonlat([apply_2d(p, a[None])[0], apply_2d(p, b[None])[0]]) for a, b in facades.segs]

        pts = np.vstack([m.pts for m in models])
        shown = np.flatnonzero((np.concatenate([m.track for m in models]) >= self.cfg.min_track)
                               & (s * pts[:, 2] + self.camera_height > 0.3))
        keep = np.random.default_rng(0).choice(shown, min(len(shown), 4000), replace=False) if len(shown) else shown
        rgb = np.vstack([m.rgb for m in models])
        plan = {
            "cameras_aligned": [lonlat(apply_2d(params, m.cams[:, :2])) for m in models],
            "cameras_mapillary": lonlat(start.gps), "raw_gps": lonlat(start.raw),
            "segments_aligned": segments(params), "segments_mapillary": segments(p0),
            "points_aligned": lonlat(apply_2d(params, pts[keep, :2])), "points_mapillary": lonlat(apply_2d(p0, pts[keep, :2])),
            "points_rgb": (rgb[keep] * 255).round().astype(int).tolist(),
            "walls": [lonlat([a, b]) for a, b in zip(walls.VA, walls.VB)],
            "facade_dist_aligned": np.round(d_after, 3).tolist(), "facade_dist_mapillary": np.round(d_before, 3).tolist(),
        }
        return {"transforms": transforms, "stats": stats, "plan": plan,
                "enu_origin_lon_lat": [start.enu.lon0, start.enu.lat0], "camera_height_m": self.camera_height}


# ----------------------------------------------------------------------------- corrected cameras

def camera_poses(transform, enu, camera_height, roi_frame):
    """Pose and calibration of every registered frame of one aligned model: {file: pose}."""
    model_txt = Path(transform["model"])
    cam_model, w, h, K, dist = read_camera(model_txt)
    a, s, t = transform["angle_rad"], transform["scale_m_per_unit"], np.array(transform["translation_m"])
    R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    origin, basis = np.array(transform["colmap_origin"]), np.array(transform["basis"])
    R_world = np.block([[R2, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]) @ basis
    intrinsics = {"camera_model": cam_model, "width": w, "height": h,
                  "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                  "k1": float(dist[0]), "k2": float(dist[1]), "p1": float(dist[2]), "p2": float(dist[3]),
                  "lens_mm_36mm_sensor": float(K[0, 0]) * 36.0 / w}
    poses = {}
    for name, C, R_wc in zip(*load_cameras(model_txt)):
        L = basis @ (C - origin)
        lon, lat = enu.lonlat(*(s * R2 @ L[:2] + t))
        height = float(s * L[2] + camera_height)
        R = R_world @ R_wc  # OpenCV camera axes (x right, y down, z forward) in ENU
        fwd = R[:, 2]
        poses[name] = {
            "lon": lon, "lat": lat, "height_m": height,
            "enu_m": [round(float(v), 4) for v in (*roi_frame.xy(lon, lat), height)],
            "rotation_cam_to_enu": np.round(R, 8).tolist(),
            "blender_matrix": np.round(R @ AXIS_FLIP, 8).tolist(),
            "heading_deg": math.degrees(math.atan2(fwd[0], fwd[1])) % 360,
            "pitch_deg": math.degrees(math.asin(np.clip(fwd[2], -1, 1))),
            "roll_deg": math.degrees(math.atan2(R[2, 0], -R[2, 1])),  # image right side up
            "intrinsics": intrinsics,
        }
    return poses


# ----------------------------------------------------------------------------- whole ROI

def _align_leg(leg, paths, frames, buildings, camera_height, height_range, max_tie_misfit):
    """Align one leg, dropping cameras whose reconstruction is broken. Sets c["dropped"] on
    left-out cameras and leg["error"] on failure; returns the result or None."""
    lo, hi = height_range
    usable = [c for c in leg["cameras"] if c.get("model") and c.get("registered", 0) >= 10]
    for c in leg["cameras"]:
        c["dropped"] = None if c in usable else f"COLMAP registered {c.get('registered', 0)}/{c['frames']} frames"
    heights = [c["camera_height_m"] for c in usable if c.get("camera_height_m") and lo <= c["camera_height_m"] <= hi]
    height = camera_height or (float(np.median(heights)) if heights else 1.4)
    print(f"  camera height {height:.2f} m{' (given)' if camera_height else ' (measured)' if heights else ' (default)'}")

    def evidence(c):  # registered share, plus one for a plausible camera height
        h = c.get("camera_height_m")
        return c["registered"] / max(c["frames"], 1) + (1.0 if h is not None and lo <= h <= hi else 0.0)

    while usable:
        inputs = [{"label": c["folder"], "model": paths.colmap / c["model"], "frames": frames[c["folder"]],
                   "masks": paths.masks / c["folder"]} for c in usable]
        try:
            result = OSMAlignmentLayer(height).align(inputs, buildings)
        except AlignmentError as e:
            culprit = next((c for c in usable[1:] if c["folder"] == e.label), None)
            if culprit is None:
                leg["error"] = str(e)
                print(f"  not aligned: {e}")
                return None
            culprit["dropped"] = str(e)
            print(f"  {culprit['folder']} dropped: {e}")
            usable.remove(culprit)
            continue
        # One rig's cameras share a path: a large time-tie misfit means a broken
        # reconstruction. Drop the less trustworthy camera of the pair and realign.
        bad = [t for t in result["stats"]["time_ties"] if t["path_misfit_rms_m"] > max_tie_misfit]
        if not bad:
            return result
        worst = max(bad, key=lambda t: t["path_misfit_rms_m"])
        other = next(c for c in usable if c["folder"] == worst["label"])
        drop = min((usable[0], other), key=evidence)
        drop["dropped"] = (f"time tie misfit {worst['path_misfit_rms_m']:.1f} m with "
                           f"{(other if drop is usable[0] else usable[0])['folder']}")
        print(f"  {drop['folder']} dropped ({drop['dropped']})")
        usable.remove(drop)
    return None


def _sequence_json(seq, frames, legs, poses, ref, max_leg_error):
    """The corrected-cameras document of one sequence."""
    folder, reasons = seq["folder"], {}
    for leg in legs:
        for c in leg["cameras"]:
            if c["folder"] == folder:
                why = c.get("dropped") or (None if leg.get("result") else leg.get("error", "leg not aligned"))
                for f in c["frame_files"]:
                    reasons.setdefault(f, f"{leg['id']}: {why or 'not registered by COLMAP'}")
    records = []
    for name, fm in sorted(frames.items()):
        leg_id, pipe, fit, reliable = poses.get((folder, name), (None, None, None, False))
        raw = fm.get("gps") or [None, None]
        rec = {"file": name, "id": fm["id"], "captured_at": fm["captured_at"], "leg": leg_id,
               "aligned": pipe is not None, "reliable": bool(reliable),
               "leg_facade_to_wall_median_m": None if fit is None else round(fit, 3), "pipeline": pipe,
               "raw_gps": {"lon": raw[0], "lat": raw[1], "compass_deg": fm.get("compass_angle")},
               "mapillary_computed": {"lon": fm["lon"], "lat": fm["lat"], "compass_deg": fm.get("computed_compass_angle"),
                                      "rotation_world_to_cam_rodrigues": fm.get("computed_rotation")}}
        if pipe:
            p = ref.xy(pipe["lon"], pipe["lat"])
            if raw[0] is not None:
                rec["raw_gps"]["offset_to_pipeline_m"] = round(math.dist(p, ref.xy(*raw)), 3)
            rec["mapillary_computed"]["offset_to_pipeline_m"] = round(math.dist(p, ref.xy(fm["lon"], fm["lat"])), 3)
            if fm.get("compass_angle") is not None:
                rec["raw_gps"]["heading_diff_to_pipeline_deg"] = round(wrap_deg(pipe["heading_deg"] - fm["compass_angle"]), 2)
        else:
            rec["not_aligned_reason"] = reasons.get(
                name, "not in a straight leg (trimmed at a turn or curve, or in a stretch too short to reconstruct)")
        records.append(rec)
    aligned = [r for r in records if r["aligned"]]
    off = [r["raw_gps"]["offset_to_pipeline_m"] for r in aligned if "offset_to_pipeline_m" in r["raw_gps"]]
    return {
        "description": "Corrected camera poses from the alignment pipeline (COLMAP aligned to OSM buildings). "
                       "pipeline.lon/lat: camera position; height_m: above the road; enu_m: metres east/north/up "
                       "from enu_origin_lon_lat; rotation_cam_to_enu: columns are the camera's x (right), "
                       "y (down) and z (viewing direction) axes in ENU; blender_matrix: the same for a "
                       "Blender camera (-z forward, +y up); intrinsics: COLMAP calibration of the "
                       "downloaded image (pixels). reliable: the camera's leg fits the OSM walls "
                       f"within {max_leg_error} m (median facade-to-wall, leg_facade_to_wall_median_m) "
                       "and better than Mapillary's positions; treat other aligned cameras with care.",
        **{k: seq[k] for k in ("sequence", "folder", "line", "role", "camera", "view_deg", "date")},
        "enu_origin_lon_lat": [ref.lon0, ref.lat0],
        "summary": {"frames": len(records), "aligned": len(aligned), "reliable": sum(r["reliable"] for r in aligned),
                    "raw_gps_to_pipeline_m": {"median": float(np.median(off)), "max": float(np.max(off))} if off else None,
                    "legs": sorted({r["leg"] for r in aligned})},
        "cameras": records,
    }


def _flags(leg, results_dir):
    """Things to check about one leg, as short messages."""
    res = leg.get("result")
    if not res:
        return [f"{leg['id']}: not aligned ({leg.get('error', 'no usable camera')})"]
    st, flags = res["stats"], []
    fit = st["facade_to_wall_aligned"]["all"]["median_m"]
    if not leg["reliable"]:
        flags.append(f"{leg['id']}: UNRELIABLE (median facade-to-wall {fit:.2f} m); its cameras are marked reliable=false")
    elif fit > 1.0:
        flags.append(f"{leg['id']}: median facade-to-wall {fit:.2f} m; check {results_dir / 'legs' / (leg['id'] + '_alignment.png')}")
    if not 0.85 <= st["scale_vs_mapillary"] <= 1.18:
        flags.append(f"{leg['id']}: scale x{st['scale_vs_mapillary']:.2f} vs Mapillary; check the calibration")
    if st["cameras_inside_buildings"]:
        flags.append(f"{leg['id']}: {st['cameras_inside_buildings']} cameras inside buildings")
    if not st["facades_both_sides"]:
        flags.append(f"{leg['id']}: facades on one side only, scale held to Mapillary's track")
    for c in leg["cameras"]:
        if c.get("dropped"):
            flags.append(f"{leg['id']}: {c['folder']} left out ({c['dropped']})")
        if c.get("merged_model"):
            flags.append(f"{leg['id']}: {c['folder']}: COLMAP split it and the pieces were merged; check for a seam")
    return flags


def align_roi(roi_dir, camera_height=None, height_range=(0.9, 3.0), max_tie_misfit=5.0, max_leg_error=2.0):
    """Align every leg of the ROI and write the corrected cameras per sequence and the images.

    A leg is reliable when its facades end up within max_leg_error (median, m) of the OSM
    walls and closer than with Mapillary's positions alone; every camera records it."""
    paths = RoiPaths(roi_dir)
    if not paths.legs_json.exists():
        raise SystemExit(f"{paths.legs_json} not found: run the colmap stage first")
    legs = json.loads(paths.legs_json.read_text())["legs"]
    sequences = paths.sequences()
    frames = {s["folder"]: paths.frames(s["folder"]) for s in sequences}
    ref = LocalFrame(*json.loads(paths.roi_json.read_text())["lines"][0][0])
    for d in (paths.alignment / "legs", paths.results / "legs"):
        d.mkdir(parents=True, exist_ok=True)

    # One Overpass request for the whole ROI.
    lons = [f["lon"] for fr in frames.values() for f in fr.values()]
    lats = [f["lat"] for fr in frames.values() for f in fr.values()]
    pad = DEFAULTS["margin"] + DEFAULTS["search_m"] + 40
    buildings = fetch_buildings((min(lons) - pad / ref.kx, min(lats) - pad / ref.ky,
                                 max(lons) + pad / ref.kx, max(lats) + pad / ref.ky))
    print(f"OSM: {len(buildings)} building outlines around the ROI")

    poses, done = {}, []  # poses: (sequence folder, frame) -> (leg, pose, leg fit, reliable)
    for leg in legs:
        print(f"\n{leg['id']} ({leg['street']}):")
        result = leg["result"] = _align_leg(leg, paths, frames, buildings, camera_height, height_range, max_tie_misfit)
        if result is None:
            continue
        st = result["stats"]
        fit = st["facade_to_wall_aligned"]["all"]["median_m"]
        leg["reliable"] = fit <= max_leg_error and fit < st["facade_to_wall_mapillary_only"]["all"]["median_m"]
        leg["plan"] = result["plan"]
        (paths.alignment / "legs" / f"{leg['id']}.json").write_text(json.dumps(
            {k: result[k] for k in ("stats", "transforms", "enu_origin_lon_lat", "camera_height_m")}, indent=1))
        plot_before_after(paths.results / "legs" / f"{leg['id']}_alignment.png", [leg], ref, buildings,
                          f"{leg['id']}: {leg['street']}")
        enu = LocalFrame(*result["enu_origin_lon_lat"])
        for tr in result["transforms"]:
            for name, pose in camera_poses(tr, enu, result["camera_height_m"], ref).items():
                # A frame in two neighbouring legs keeps the pose from the better-fitting leg.
                key = (tr["label"], name)
                if key not in poses or fit < poses[key][2]:
                    poses[key] = (leg["id"], pose, fit, leg["reliable"])
        done.append(leg)

    records = []
    for seq in sequences:
        doc = _sequence_json(seq, frames[seq["folder"]], legs, poses, ref, max_leg_error)
        (paths.alignment / f"{seq['folder']}.json").write_text(json.dumps(doc, indent=1))
        records += doc["cameras"]
        s = doc["summary"]
        print(f"\n{seq['folder']}: {s['aligned']}/{s['frames']} frames aligned ({s['reliable']} reliable)"
              + (f", raw GPS to pipeline median {s['raw_gps_to_pipeline_m']['median']:.1f} m" if s["raw_gps_to_pipeline_m"] else ""))
    if not done:
        raise SystemExit("No leg could be aligned; see the messages above")
    plot_before_after(paths.results / "alignment_overview.png", done, ref, buildings,
                      " -> ".join(f"{leg['id']}: {leg['street']}" for leg in done))
    plot_gps_vs_pipeline(paths.results / "raw_gps_vs_pipeline.png", records, ref, buildings)

    # Summary.
    off = [r["raw_gps"]["offset_to_pipeline_m"] for r in records if "offset_to_pipeline_m" in r["raw_gps"]]
    d_before = np.concatenate([leg["plan"]["facade_dist_mapillary"] for leg in done])
    d_after = np.concatenate([leg["plan"]["facade_dist_aligned"] for leg in done])
    leg_rows = []
    for leg in legs:
        row = {"leg": leg["id"], "street": leg["street"], "aligned": bool(leg["result"]), "reliable": bool(leg.get("reliable")),
               "cameras": [{k: c.get(k) for k in ("folder", "frames", "registered", "camera_height_m", "calibration",
                                                  "colmap_pieces", "dropped")} for c in leg["cameras"]]}
        if leg["result"]:
            st = leg["result"]["stats"]
            row.update(facade_to_wall_median_m={"mapillary_only": st["facade_to_wall_mapillary_only"]["all"]["median_m"],
                                                "aligned": st["facade_to_wall_aligned"]["all"]["median_m"]},
                       aligned_within_1m=st["facade_to_wall_aligned"]["all"]["within_1m"],
                       scale_vs_mapillary=st["scale_vs_mapillary"], facades_both_sides=st["facades_both_sides"],
                       camera_height_m=leg["result"]["camera_height_m"])
        leg_rows.append(row)
    summary = {"roi": paths.name, "legs": leg_rows, "flags": [f for leg in legs for f in _flags(leg, paths.results)],
               "frames": len(records), "frames_aligned": sum(r["aligned"] for r in records),
               "frames_reliable": sum(r["reliable"] for r in records),
               "facade_to_wall_median_m": {"mapillary_only": float(np.median(d_before)), "aligned": float(np.median(d_after))},
               "raw_gps_to_pipeline_median_m": float(np.median(off)) if off else None,
               "outputs": [f"{s['folder']}.json" for s in sequences], "images": str(paths.results)}
    paths.summary_json.write_text(json.dumps(summary, indent=1))

    print(f"\n=== {paths.name}: {summary['frames_aligned']}/{summary['frames']} frames aligned, "
          f"{summary['frames_reliable']} reliable ===")
    for row in leg_rows:
        fw = row.get("facade_to_wall_median_m")
        print(f"  {row['leg']:6s} {row['street'][:28]:28s} " + (
            f"facade-to-wall {fw['mapillary_only']:5.2f} -> {fw['aligned']:5.2f} m "
            f"({100 * row['aligned_within_1m']:.0f}% within 1 m){'' if row['reliable'] else '  UNRELIABLE'}"
            if fw else "not aligned"))
    print(f"  overall facade-to-wall median {np.median(d_before):.2f} -> {np.median(d_after):.2f} m"
          + (f"; raw GPS to pipeline median {np.median(off):.1f} m" if off else ""))
    print("  flags: " + ("\n         ".join(summary["flags"]) or "none"))
    print(f"Corrected cameras in {paths.alignment}; images in {paths.results}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roi_dir", type=Path, help="ROI folder (rois/mapillary_roi_N) after the colmap stage")
    parser.add_argument("--camera-height", type=float, help="Camera height above the road (m); default: measured per leg")
    parser.add_argument("--height-range", type=float, nargs=2, default=[0.9, 3.0], help="Plausible camera heights (m)")
    parser.add_argument("--max-tie-misfit", type=float, default=5.0,
                        help="Drop a rig camera whose path disagrees with the others by more than this (m)")
    parser.add_argument("--max-leg-error", type=float, default=2.0,
                        help="Legs fitting the walls worse than this (median m) are marked unreliable")
    args = parser.parse_args()
    align_roi(args.roi_dir, args.camera_height, args.height_range, args.max_tie_misfit, args.max_leg_error)


if __name__ == "__main__":
    main()
