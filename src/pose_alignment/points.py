"""The alignment's sparse point cloud on the map, for viewers.

Each leg camera's COLMAP points go through that camera's fit to the OSM footprints
(data/alignment/<leg>_<folder>/transform.json, as plot_fit.py draws them): levelled, then
each piece's similarity, giving ENU metres from the ROI's origin and a height above the
road. Only reliable legs, and points seen by at least MIN_TRACK cameras, between 3 m below
the road and MAX_HEIGHT_M above it, and within MAX_DIST_M of the leg's cameras (far points
are poorly triangulated).
"""

from __future__ import annotations

import json
import math

import numpy as np

from .geo_common import LocalFrame, RoiPaths, load_cameras, load_points

MIN_TRACK = 3
MAX_HEIGHT_M = 100.0
MAX_DIST_M = 100.0


def leg_points(folder_dir):
    """(ENU xyz (N, 3), with z the height above the road; rgb (N, 3) uint8; ENU origin
    (lon, lat)) of one aligned leg camera."""
    from .align import point_pieces  # (align imports matplotlib)

    tf = json.loads((folder_dir / "transform.json").read_text())
    origin, basis = np.array(tf["origin"]), np.array(tf["basis"])
    pieces = tf.get("pieces") or [
        {
            "images": None,
            "angle_rad": tf["angle_rad"],
            "scale_m_per_unit": tf["scale_m_per_unit"],
            "translation_m": tf["translation_m"],
        }
    ]
    names, C, _ = load_cameras(tf["model"])
    xyz, rgb, track = load_points(tf["model"])
    piece_of = {n: k for k, pc in enumerate(pieces) for n in (pc["images"] or names)}
    idx = (
        point_pieces(tf["model"], names, piece_of)
        if len(pieces) > 1
        else np.zeros(len(xyz), int)
    )

    def to_enu(X, which):
        L = (X - origin) @ basis.T
        out = np.empty_like(L)
        for k, pc in enumerate(pieces):
            s, a = pc["scale_m_per_unit"], pc["angle_rad"]
            Rz = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            m = which == k
            out[m] = np.c_[
                s * L[m, :2] @ Rz.T + np.array(pc["translation_m"]),
                s * (L[m, 2] - tf["ground_z"]),
            ]
        return out

    out = to_enu(xyz, idx)
    cams = to_enu(C, np.array([piece_of[n] for n in names]))
    from scipy.spatial import cKDTree

    near = cKDTree(cams[:, :2]).query(out[:, :2])[0] <= MAX_DIST_M
    keep = (track >= MIN_TRACK) & (out[:, 2] > -3.0) & (out[:, 2] < MAX_HEIGHT_M) & near
    colours = (np.clip(rgb[keep], 0, 1) * 255).astype(np.uint8)
    return out[keep], colours, tf["enu_origin_lon_lat"]


def aligned_points(roi_dir, max_points=200_000, seed=0):
    """{"lon", "lat", "height_m" (above the road), "rgb"} of the reliable legs' points,
    subsampled to at most max_points; None when there are none."""
    paths = RoiPaths(roi_dir)
    if not paths.summary_json.exists():
        return None
    lon, lat, h, rgb = [], [], [], []
    for e in json.loads(paths.summary_json.read_text())["legs"]:
        d = paths.alignment / f"{e['leg']}_{e['folder']}"
        if not (
            e.get("aligned") and e.get("reliable") and (d / "transform.json").exists()
        ):
            continue
        xyz, colours, origin = leg_points(d)
        frame = LocalFrame(*origin)
        lo, la = frame.lonlat(xyz[:, 0], xyz[:, 1])
        lon.append(np.asarray(lo))
        lat.append(np.asarray(la))
        h.append(xyz[:, 2])
        rgb.append(colours)
    if not lon:
        return None
    lon_a, lat_a, h_a, rgb_a = (np.concatenate(v) for v in (lon, lat, h, rgb))
    if len(lon_a) > max_points:
        pick = np.random.default_rng(seed).choice(len(lon_a), max_points, replace=False)
        lon_a, lat_a, h_a, rgb_a = lon_a[pick], lat_a[pick], h_a[pick], rgb_a[pick]
    return {"lon": lon_a, "lat": lat_a, "height_m": h_a, "rgb": rgb_a}
