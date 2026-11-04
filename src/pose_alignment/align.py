"""Stage 6 (align): ground-plane projection and alignment to the OpenStreetMap footprints.

Per leg and sequence model:
  1. Level with GeoCalib's gravity: each camera's down direction (GeoCalib, camera frame) turned
     into the model's frame by that camera's rotation; their robust mean is the model's down.
     Plan view: x right of travel, y along it, z up.
  2. Ground plane: the median height of the road points (segmentation) near the camera path.
  3. Building points: points labelled Building in most of their observations, seen in 3+
     frames, 2.5-40 m above the ground, thinned to one per 0.5 m cell in plan view.
  4. GPS as a prior only: a 2D similarity of the levelled camera path onto Mapillary's
     positions gives the starting scale, heading and position (and weak priors in the fit).
  5. Fit: a global search over heading (+-15 deg), scale and position (+-20 m) for the placement
     whose building points lie closest to the OSM footprint edges (cameras inside buildings
     penalised), then a robust (Cauchy) least-squares refinement of point-to-edge distances,
     as Alignment_Pipeline's "points" method. One 2D similarity per leg, or per piece when the leg
     was reconstructed with --split-weak: the leg is cut where reconstruct.py dropped weakly
     anchored frames (a gap the images do not hold together; a frame
     Mapillary skipped is not a cut), pieces under 8 cameras join a neighbour, and each 3D point
     goes with the piece whose cameras observe it most. stats.json reports each cut's aligned
     length next to the GPS distance (a yardstick only) and the model's neighbouring speed.
     Rig cameras (reconstruct.py's rigs: one model for all cameras of a leg) are placed together:
     the reference camera's fit uses the building points of every rig camera, levelled with its
     own gravity and placed from its own GPS path, and the other cameras get that transform
     instead of a fit of their own (roi_11's right-side camera, fitted alone, locked onto the
     wrong buildings 18-26 m off while the left cameras of the same model fitted to 0.9 m).
     Linked legs (reconstruct.py's leg overlap: a leg runs on into the next one's first frames):
     after every leg is fitted on its own, the fits of legs sharing frames are refined together,
     each still on its own building points, plus the condition that the shared cameras land in the
     same place from both legs (LINK_SIGMA). That lines the legs up end to end and carries their
     scales across (a leg whose walls leave it free along the street is held by its neighbours).
     All legs are placed in one ENU frame (the route's first point).
  6. Reliable when the building points end within --max-error (2 m) of the footprints (median),
     at most 20% of the cameras end inside buildings, and no camera lies behind an earlier one
     along the leg (a folded reconstruction; stats.json lists them as cameras_out_of_order).
Writes data/alignment/: <leg>_<folder>/{stats.json, transform.json, plan.png}, <folder>.json (aligned
cameras: lon, lat, height, heading, pitch, roll), summary.json; results/overview.png and
fit_<leg>_<folder>.png (plot_fit.py: the sparse cloud on the footprints, plan and 3D views).

Usage: python -m pose_alignment.align outputs/alignment/mapillary_roi_2
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from .geo_common import (
    MASK_BUILDING,
    MASK_ROAD,
    LocalFrame,
    RoiPaths,
    apply_2d,
    label_points,
    load_cameras,  # noqa: E402
    load_points,
    read_camera,
    segment_distances,
    similarity_2d,
)
from .osm import WallMap, building_heights, fetch_buildings, side_of_path  # noqa: E402

OPT = {
    "min_track": 3,
    "min_height": 2.5,
    "max_height": 40.0,
    "class_fraction": 0.5,
    "point_grid": 0.5,
    "search_points": 800,
    "search_deg": 15.0,
    "search_m": 20.0,
    "shift_prior": 15.0,
    "match_dist": 6.0,
    "wall_scale": 0.75,
    "gps_sigma": 25.0,
    "margin": 60.0,
    "default_camera_height": 1.6,
    # scale vs the GPS placement's: searched range, prior std (log) and refinement freedom, with
    # building points on both sides of the street (scale observable) / on one side only
    "scale_range_both": (0.8, 1.4),
    "scale_range_one": (0.8, 1.4),
    "scale_sigma_both": 0.3,
    "scale_sigma_one": 0.1,
    "refine_scale": 1.25,
}


_LABELS = {}


def labels(model, mask_dir, value):
    """label_points, kept per model, masks and value: a rig's cameras are labelled again for the
    reference camera's pooled fit, and linked legs are placed twice."""
    key = (
        str(model),
        str(mask_dir),
        value,
        (Path(model) / "points3D.txt").stat().st_mtime_ns,
    )
    if key not in _LABELS:
        _LABELS[key] = label_points(model, mask_dir, value, OPT["class_fraction"])
    return _LABELS[key]


def level_with_gravity(names, C, R, gc):
    """(origin, basis rows [right, forward, up], gravity spread deg) from GeoCalib's per-frame
    gravity. R: world-from-camera rotations."""
    d = np.array(
        [R[i] @ -np.asarray(gc["frames"][n]["up_cam"]) for i, n in enumerate(names)]
    )
    keep = np.ones(len(d), bool)
    for _ in range(3):
        m = d[keep].mean(axis=0)
        m /= np.linalg.norm(m)
        ang = np.degrees(np.arccos(np.clip(d @ m, -1, 1)))
        keep = ang < max(3 * np.median(ang), 2.0)
    up = -m
    fwd = C[-1] - C[0]
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    return C[0], np.stack([right, fwd, up]), float(np.median(ang)), int((~keep).sum())


def heading_pitch_roll(R_enu):
    """Camera (world-from-camera rotation in east/north/up) -> heading (deg clockwise from north),
    pitch (deg, up positive), roll (deg)."""
    z, x = R_enu[:, 2], R_enu[:, 0]
    heading = math.degrees(math.atan2(z[0], z[1])) % 360
    pitch = math.degrees(math.asin(np.clip(z[2], -1, 1)))
    roll = math.degrees(math.asin(np.clip(x[2], -1, 1)))
    return heading, pitch, roll


def split_pieces(names, meta, dropped=(), min_cameras=8):
    """Camera index groups of a leg, cut only where frames were dropped as weakly anchored
    (reconstruct.py's dropped_frames: a gap the images do not hold together); a time gap alone
    (a frame Mapillary skipped) is not a cut. Pieces under min_cameras join their nearer neighbour."""
    t = np.array([meta[n]["captured_at_ms"] for n in names]) / 1000
    cuts = [
        i
        for i in range(1, len(names))
        if any(names[i - 1] < d < names[i] for d in dropped)
    ]
    pieces = [list(range(a, b)) for a, b in zip([0] + cuts, cuts + [len(names)])]
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for k, pc in enumerate(pieces):
            if len(pc) < min_cameras:
                if k == 0:
                    j = 1
                elif k == len(pieces) - 1:
                    j = k - 1
                else:  # the neighbour across the shorter time gap
                    j = (
                        k - 1
                        if t[pc[0]] - t[pieces[k - 1][-1]]
                        <= t[pieces[k + 1][0]] - t[pc[-1]]
                        else k + 1
                    )
                pieces[min(j, k)] = sorted(pieces[j] + pc)
                del pieces[max(j, k)]
                changed = True
                break
    return pieces


def point_pieces(model_txt, names, piece_of):
    """Per 3D point (points3D.txt order): the piece whose cameras observe it most."""
    raw = [
        row
        for row in (Path(model_txt) / "images.txt").read_text().splitlines()
        if not row.startswith("#")
    ]
    ids, i = {}, 0
    while i < len(raw):
        if raw[i].strip():
            q = raw[i].split()
            ids[int(q[0])] = piece_of[Path(q[9]).name]
            i += 2
        else:
            i += 1
    out = []
    for row in (Path(model_txt) / "points3D.txt").read_text().splitlines():
        if row and not row.startswith("#"):
            img = [int(v) for v in row.split()[8::2]]
            votes = np.bincount([ids[k] for k in img if k in ids], minlength=1)
            out.append(int(np.argmax(votes)))
    return np.array(out, int)


def fit_to_footprints(bpts, Lc, gps, A, B, wmap, both_sides):
    """One 2D similarity [angle, log scale, tx, ty] placing the building points bpts (levelled
    plan coordinates) closest to the footprint edges A->B: GPS-seeded global search, then a
    robust refinement with weak priors on the mean camera position and the GPS scale.
    Returns (params, params_gps_only)."""
    s_gps, ang0, t0 = similarity_2d(Lc, gps)
    params0 = np.array([ang0, math.log(s_gps), *t0])
    scale_sigma = OPT["scale_sigma_both"] if both_sides else OPT["scale_sigma_one"]
    scale_range = OPT["scale_range_both"] if both_sides else OPT["scale_range_one"]
    search = (
        bpts
        if len(bpts) <= OPT["search_points"]
        else bpts[
            np.random.default_rng(0).choice(
                len(bpts), OPT["search_points"], replace=False
            )
        ]
    )
    cam_mid, gps_mid = Lc.mean(axis=0), gps.mean(axis=0)
    offsets = np.arange(-OPT["search_m"], OPT["search_m"] + 1e-9, 2.0)
    shift_cost = (
        0.5
        * (np.hypot(*np.meshgrid(offsets, offsets, indexing="ij")) / OPT["shift_prior"])
        ** 2
    )
    best = (np.inf, None)
    for dang in np.radians(
        np.arange(-OPT["search_deg"], OPT["search_deg"] + 1e-9, 1.0)
    ):
        ang = params0[0] + dang
        Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        for sc in s_gps * np.exp(np.linspace(*np.log(scale_range), 15)):
            t = gps_mid - sc * Rm @ cam_mid
            prm = [ang, math.log(sc), *t]
            base, cams = apply_2d(prm, search), apply_2d(prm, Lc)
            for ix, dx in enumerate(offsets):
                shift = np.stack([np.full_like(offsets, dx), offsets], axis=1)[:, None]
                score = np.minimum(wmap.distance(base[None] + shift), 3.0).mean(axis=1)
                score += (
                    5.0 * wmap.inside(cams[None] + shift).mean(axis=1) + shift_cost[ix]
                )
                j = int(np.argmin(score))
                if score[j] < best[0]:
                    best = (
                        score[j],
                        np.array([ang, math.log(sc), t[0] + dx, t[1] + offsets[j]]),
                    )
    params = best[1]
    wdir = (B - A) / np.linalg.norm(B - A, axis=1)[:, None]
    wnrm = np.stack([-wdir[:, 1], wdir[:, 0]], axis=1)
    lower = [
        params[0] - math.radians(5),
        params[1] - math.log(OPT["refine_scale"]),
        params[2] - 5,
        params[3] - 5,
    ]
    upper = [
        params[0] + math.radians(5),
        params[1] + math.log(OPT["refine_scale"]),
        params[2] + 5,
        params[3] + 5,
    ]
    for _ in range(10):
        d = segment_distances(apply_2d(params, bpts), A, B)
        k = np.argmin(d, axis=1)
        matched = d[np.arange(len(bpts)), k] < OPT["match_dist"]
        S, Ad, Nd = bpts[matched], A[k[matched]], wnrm[k[matched]]
        sw = 1.0 / np.sqrt(
            1.0
            + (
                np.abs(np.sum((apply_2d(params, S) - Ad) * Nd, axis=1))
                / OPT["wall_scale"]
            )
            ** 2
        )
        n = max(int(matched.sum()), 1)

        def residuals(prm):
            walls = (
                sw * np.sum((apply_2d(prm, S) - Ad) * Nd, axis=1) / OPT["wall_scale"]
            )
            g = (
                math.sqrt(n)
                * (apply_2d(prm, Lc).mean(axis=0) - gps_mid)
                / OPT["gps_sigma"]
            )
            sc = math.sqrt(n) * (prm[1] - params0[1]) / scale_sigma
            return np.r_[walls, g, sc]

        params = least_squares(
            residuals,
            np.clip(params, lower, upper),
            bounds=(lower, upper),
            x_scale=[0.01, 0.01, 1, 1],
        ).x
    return params, params0


def rig_building_points(p, cam, origin, basis, s_gps):
    """A rig camera's usable building points (as align_camera's) in another camera's levelled
    plan frame (the same model world frame): (N, 2)."""
    model = p.reconstruction / cam["model"]
    _, C, _ = load_cameras(model)
    xyz, _, track = load_points(model)
    L, Lc = (xyz - origin) @ basis.T, (C - origin) @ basis.T
    masks = p.masks / cam["folder"]
    road, _ = labels(model, masks, MASK_ROAD)
    near = (
        np.min(np.linalg.norm(L[:, None, :2] - Lc[None, ::3, :2], axis=2), axis=1)
        * s_gps
        < 25
    )
    ground_z = (
        float(np.median(L[road & near, 2]))
        if (road & near).sum() >= 30
        else float(np.median(Lc[:, 2])) - OPT["default_camera_height"] / s_gps
    )
    building, _ = labels(model, masks, MASK_BUILDING)
    h_m = (L[:, 2] - ground_z) * s_gps
    sel = np.flatnonzero(
        building
        & (track >= OPT["min_track"])
        & (h_m > OPT["min_height"])
        & (h_m < OPT["max_height"])
    )
    _, first = np.unique(
        np.floor(L[sel, :2] / (OPT["point_grid"] / s_gps)).astype(np.int64),
        axis=0,
        return_index=True,
    )
    return L[sel[np.sort(first)], :2]


def align_camera(
    p, leg, cam, buildings, out_dir, max_error, shared=None, pool=(), enu=None
):
    """shared: a rig's placement from its reference camera ({origin, basis, enu, fits, fits0}; this
    camera's model is in the same world frame), used instead of levelling and fitting this camera
    on its own. pool: the rig's other cameras, whose building points join this camera's in the fit.
    enu: the plan frame (default: this camera's first frame)."""
    folder = cam["folder"]
    meta = p.frames(folder)
    gc = json.loads(p.calibration_json(folder).read_text())
    model = p.reconstruction / cam["model"]
    names, C, R = load_cameras(model)
    origin, basis, g_spread, g_out = level_with_gravity(names, C, R, gc)
    if shared:
        origin, basis = shared["origin"], shared["basis"]
    xyz, rgb, track = load_points(model)
    L = (xyz - origin) @ basis.T
    Lc = (C - origin) @ basis.T

    # GPS (Mapillary's positions) as the starting placement and the leg's metric scale
    enu = (
        shared["enu"]
        if shared
        else enu or LocalFrame(meta[names[0]]["lon"], meta[names[0]]["lat"])
    )
    gps = np.array([enu.xy(meta[n]["lon"], meta[n]["lat"]) for n in names])
    s_gps = similarity_2d(Lc[:, :2], gps)[0]

    # ground plane from the road points near the path
    masks = p.masks / folder
    road, _ = labels(model, masks, MASK_ROAD)
    near = (
        np.min(np.linalg.norm(L[:, None, :2] - Lc[None, ::3, :2], axis=2), axis=1)
        * s_gps
        < 25
    )
    cam_z = float(np.median(Lc[:, 2]))
    if (road & near).sum() >= 30:
        ground_z = float(np.median(L[road & near, 2]))
        ground_src = f"{int((road & near).sum())} road points"
    else:
        ground_z = cam_z - OPT["default_camera_height"] / s_gps
        ground_src = f"default camera height {OPT['default_camera_height']} m"
    cam_height = (cam_z - ground_z) * s_gps

    building, _ = labels(model, masks, MASK_BUILDING)
    h_m = (L[:, 2] - ground_z) * s_gps
    usable = (
        building
        & (track >= OPT["min_track"])
        & (h_m > OPT["min_height"])
        & (h_m < OPT["max_height"])
    )

    # pieces: the leg cut where weak frames were dropped; each gets its own fit
    pieces = split_pieces(names, meta, cam.get("dropped_frames", {}))
    piece_of = {names[i]: k for k, pc in enumerate(pieces) for i in pc}
    cam_piece = np.array([piece_of[n] for n in names])
    pt_piece = (
        point_pieces(model, names, piece_of)
        if len(pieces) > 1
        else np.zeros(len(xyz), int)
    )

    lo = gps.min(axis=0) - OPT["margin"] - OPT["search_m"]
    hi = gps.max(axis=0) + OPT["margin"] + OPT["search_m"]
    footprints = []
    for tags, ring in buildings:
        xy = np.array([enu.xy(lon, lat) for lon, lat in ring])
        if np.all(xy.max(axis=0) >= lo) and np.all(xy.min(axis=0) <= hi):
            base, top, _ = building_heights(tags, 3.5, 3)
            footprints.append((xy, base, top))
    if not footprints:
        return {"aligned": False, "reason": "no OSM buildings near the track"}
    A = np.vstack([xy[:-1] for xy, _, _ in footprints])
    B = np.vstack([xy[1:] for xy, _, _ in footprints])
    ok = np.linalg.norm(B - A, axis=1) >= 0.5
    A, B = A[ok], B[ok]
    wmap = WallMap(A, B, footprints, lo, hi)

    # building points thinned to one per point_grid cell, per piece
    sel = np.flatnonzero(usable)
    _, first = np.unique(
        np.c_[
            np.floor(L[sel, :2] / (OPT["point_grid"] / s_gps)).astype(np.int64),
            pt_piece[sel],
        ],
        axis=0,
        return_index=True,
    )
    sel = sel[np.sort(first)]
    if len(sel) < 30:
        return {"aligned": False, "reason": f"only {len(sel)} usable building points"}
    bpts, b_piece = L[sel, :2], pt_piece[sel]
    pooled = (
        np.vstack([rig_building_points(p, c, origin, basis, s_gps) for c in pool])
        if pool
        else np.zeros((0, 2))
    )
    right_share = float(np.mean(side_of_path(np.vstack([bpts, pooled]), [Lc[:, :2]])))
    both_sides = min(right_share, 1 - right_share) >= 0.2

    fits, fits0, piece_stats = [], [], []
    p.results.mkdir(parents=True, exist_ok=True)
    for k, pc in enumerate(pieces):
        if shared:  # a rig camera: the reference camera's placement
            fits.append(np.asarray(shared["fits"][min(k, len(shared["fits"]) - 1)]))
            fits0.append(np.asarray(shared["fits0"][min(k, len(shared["fits0"]) - 1)]))
            continue
        mine = b_piece == k
        if (
            mine.sum() < 30
        ):  # too few points of its own: fit with the whole leg's points
            mine = np.ones(len(bpts), bool)
        prm, prm0 = fit_to_footprints(
            np.vstack([bpts[mine], pooled]), Lc[pc, :2], gps[pc], A, B, wmap, both_sides
        )
        fits.append(prm)
        fits0.append(prm0)

    def place(params_list, XY, idx):
        out = np.empty_like(XY)
        for k, prm in enumerate(params_list):
            m = idx == k
            out[m] = apply_2d(prm, XY[m])
        return out

    def ptf(params_list, mask=None):
        P = place(params_list, bpts, b_piece)
        dd = np.minimum(
            segment_distances(P if mask is None else P[mask], A, B).min(axis=1), 50.0
        )
        return {
            "points": int(len(dd)),
            "median_m": float(np.median(dd)),
            "within_1m": float(np.mean(dd < 1.0)),
            "within_2m": float(np.mean(dd < 2.0)),
        }

    before, after = ptf(fits0), ptf(fits)
    cams_xy = place(fits, Lc[:, :2], cam_piece)
    inside = wmap.inside(cams_xy)
    # Cameras must follow capture order along the leg: a camera placed behind an earlier one (a step
    # backwards of more than a third of a normal step) means the reconstruction folded, e.g. a
    # stretch slid back over the previous one, however well its points fit the walls
    steps = np.diff(Lc[:, 1])
    backward = [
        names[i + 1] for i in np.flatnonzero(steps < -np.median(np.abs(steps)) / 3)
    ]
    reliable = after["median_m"] <= max_error and inside.mean() <= 0.2 and not backward
    t_s = np.array([meta[n]["captured_at_ms"] for n in names]) / 1000
    for k, (pc, prm, prm0) in enumerate(zip(pieces, fits, fits0)):
        st = {
            "cameras": len(pc),
            "first": names[pc[0]],
            "last": names[pc[-1]],
            "building_points": int((b_piece == k).sum()),
            "points_to_footprint": ptf(fits, b_piece == k),
            "scale_vs_mapillary": math.exp(prm[1] - prm0[1]),
            "heading_change_deg": math.degrees(prm[0] - prm0[0]),
        }
        if k:
            # the cut: aligned distance across it against the speed on either side
            i0, i1 = pieces[k - 1][-1], pc[0]
            v = lambda a, b: np.linalg.norm(cams_xy[b] - cams_xy[a]) / (t_s[b] - t_s[a])  # noqa: E731
            v_before = (
                v(max(pieces[k - 1][0], i0 - 3), i0)
                if i0 > pieces[k - 1][0]
                else float("nan")
            )
            v_after = v(i1, min(pc[-1], i1 + 3)) if i1 < pc[-1] else float("nan")
            # GPS only as a yardstick here; the model's own speed next to a weak gap can be off too
            st["cut"] = {
                "seconds": float(t_s[i1] - t_s[i0]),
                "aligned_distance_m": float(np.linalg.norm(cams_xy[i1] - cams_xy[i0])),
                "gps_distance_m": float(np.linalg.norm(gps[i1] - gps[i0])),
                "model_speed_distance_m": float(
                    np.nanmean([v_before, v_after]) * (t_s[i1] - t_s[i0])
                ),
            }
        piece_stats.append(st)

    s = math.exp(fits[0][1])
    cam_model, cam_w, cam_h, cam_params = read_camera(model)
    # what texturing needs: the COLMAP camera (its distortion is not undone) at the frame's size
    intrinsics = {
        "model": cam_model,
        "width": cam_w,
        "height": cam_h,
        "fx": cam_params[0],
        "params": cam_params,
    }
    cams_out = []
    for i, nme in enumerate(names):
        prm = fits[cam_piece[i]]
        a = prm[0]
        Rz = np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )
        R_enu = Rz @ basis @ R[i]
        hdg, pit, rol = heading_pitch_roll(R_enu)
        lon, lat = enu.lonlat(*cams_xy[i])
        cams_out.append(
            {
                "file": nme,
                "id": meta[nme].get("id"),
                "leg": leg["id"],
                "lon": lon,
                "lat": lat,
                "height_m": float((Lc[i, 2] - ground_z) * math.exp(prm[1])),
                "heading_deg": hdg,
                "pitch_deg": pit,
                "roll_deg": rol,
                # columns: the camera's x (right), y (down), z (forward) in east/north/up
                "rotation_cam_to_enu": np.round(R_enu, 8).tolist(),
                "intrinsics": intrinsics,
                "piece": int(cam_piece[i]),
                "pano_yaw_deg": meta[nme].get("pano_yaw_deg"),
                "gps": meta[nme].get("gps"),
                "mapillary_lonlat": [meta[nme]["lon"], meta[nme]["lat"]],
                "offset_to_mapillary_m": float(np.linalg.norm(cams_xy[i] - gps[i])),
                "leg_fit_m": after["median_m"],
                "reliable": bool(reliable),
            }
        )

    n_w = np.array([len(pc) for pc in pieces], float)
    stats = {
        "leg": leg["id"],
        "folder": folder,
        "registered": len(names),
        "frames": cam["frames"],
        "gravity": {"spread_median_deg": g_spread, "outlier_cameras": g_out},
        "ground": ground_src,
        "camera_height_m": cam_height,
        "building_points": int(len(bpts)),
        "facades_both_sides": both_sides,
        "pieces": piece_stats,
        "points_to_footprint_median_m": {
            "mapillary_only": before["median_m"],
            "aligned": after["median_m"],
        },
        "points_to_footprint_within_1m": after["within_1m"],
        "points_to_footprint": {"before": before, "after": after},
        "scale_m_per_unit": s,
        "scale_vs_mapillary": float(
            np.average([st["scale_vs_mapillary"] for st in piece_stats], weights=n_w)
        ),
        "heading_change_deg": float(
            np.average([st["heading_change_deg"] for st in piece_stats], weights=n_w)
        ),
        "cameras_inside_buildings": int(inside.sum()),
        "cameras_out_of_order": backward,
        "rig_fit": (
            f"placed with the rig reference {cam['rig']}"
            if shared and cam.get("rig") not in (None, folder)
            else f"fitted with the building points of {len(pool)} other rig cameras ({len(pooled)} points)"
            if pool
            else None
        ),
        "aligned": True,
        "reliable": bool(reliable),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    (out_dir / "transform.json").write_text(
        json.dumps(
            {
                "description": "levelled L = basis @ (X_colmap - origin) (x right, y forward, z up); per piece, ENU metres "
                "from enu_origin: xy = s R(angle) L[:2] + t, height above ground = s (L[2] - ground_z). "
                "A 3D point belongs to the piece whose cameras observe it most (align.point_pieces)",
                "model": str(model),
                "origin": origin.tolist(),
                "basis": basis.tolist(),
                "ground_z": ground_z,
                "angle_rad": float(fits[0][0]),
                "scale_m_per_unit": s,
                "translation_m": fits[0][2:].tolist(),
                "pieces": [
                    {
                        "images": [names[i] for i in pc],
                        "angle_rad": float(prm[0]),
                        "scale_m_per_unit": math.exp(prm[1]),
                        "translation_m": prm[2:].tolist(),
                        "params_gps_only": prm0.tolist(),
                    }
                    for pc, prm, prm0 in zip(pieces, fits, fits0)
                ],
                "enu_origin_lon_lat": [enu.lon0, enu.lat0],
            },
            indent=1,
        )
    )

    # plan view: GPS-only placement vs fitted
    shown = (track >= OPT["min_track"]) & (h_m > 0.3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, plist, title in (
        (
            axes[0],
            fits0,
            f"GPS placement: building points to footprints {before['median_m']:.2f} m",
        ),
        (
            axes[1],
            fits,
            f"Fitted to OSM{f' in {len(pieces)} pieces' if len(pieces) > 1 else ''}: "
            f"{after['median_m']:.2f} m ({100 * after['within_1m']:.0f}% within 1 m)",
        ),
    ):
        for xy, _, _ in footprints:
            ax.fill(*xy.T, color="#e3e3e3", ec="#888", lw=0.6, zorder=0)
        ax.scatter(
            *place(plist, L[shown, :2], pt_piece[shown]).T,
            s=0.4,
            c=rgb[shown],
            zorder=1,
        )
        ax.scatter(
            *place(plist, bpts, b_piece).T,
            s=1.2,
            c="#d62728",
            zorder=2,
            label="building points",
        )
        cxy = place(plist, Lc[:, :2], cam_piece)
        for k in range(len(pieces)):
            ax.plot(
                *cxy[cam_piece == k].T,
                ".-",
                ms=3,
                lw=1,
                zorder=3,
                label=f"cameras, piece {k + 1}" if len(pieces) > 1 else "cameras",
            )
        ax.plot(*gps.T, "x", color="#555", ms=3, zorder=3, label="Mapillary positions")
        allp = np.vstack([cxy, gps])
        ax.set_xlim(allp[:, 0].min() - 40, allp[:, 0].max() + 40)
        ax.set_ylim(allp[:, 1].min() - 40, allp[:, 1].max() + 40)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"{p.name} {leg['id']}/{folder}: {len(names)} cameras, GeoCalib gravity levelling "
        f"(spread {g_spread:.1f} deg), camera height {cam_height:.2f} m, "
        f"{'reliable' if reliable else 'UNRELIABLE'}",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "plan.png", dpi=110)
    plt.close(fig)
    fitdata = None
    if (
        not shared and len(pieces) == 1
    ):  # what link_legs needs to refit this camera with its neighbours
        fitdata = {
            "bpts": np.vstack([bpts, pooled]),
            "Lc": Lc[:, :2],
            "gps": gps,
            "A": A,
            "B": B,
            "names": names,
            "params0": fits0[0],
            "scale_sigma": OPT["scale_sigma_both"]
            if both_sides
            else OPT["scale_sigma_one"],
        }
    return {
        **stats,
        "cameras_out": cams_out,
        "_shared": {
            "origin": origin,
            "basis": basis,
            "enu": enu,
            "fits": fits,
            "fits0": fits0,
        },
        "_fitdata": fitdata,
    }


LINK_SIGMA = 0.3  # m: how closely two linked legs must place the cameras they share
LINK_SHAPE_M = 1.0  # two legs' shared cameras further apart than this after the best similarity: not linked
LINK_WORSE_M, LINK_WORSE_X = (
    0.5,
    1.5,
)  # a link making a leg's wall fit worse by more than both of these is undone


def wall_terms(params, bpts, A, B):
    """The wall residual set-up of fit_to_footprints' refinement at params: (points, wall start,
    wall normal, robust weights) of the building points within match_dist of a wall."""
    wdir = (B - A) / np.linalg.norm(B - A, axis=1)[:, None]
    wnrm = np.stack([-wdir[:, 1], wdir[:, 0]], axis=1)
    d = segment_distances(apply_2d(params, bpts), A, B)
    k = np.argmin(d, axis=1)
    m = d[np.arange(len(bpts)), k] < OPT["match_dist"]
    S, Ad, Nd = bpts[m], A[k[m]], wnrm[k[m]]
    sw = 1.0 / np.sqrt(
        1.0
        + (np.abs(np.sum((apply_2d(params, S) - Ad) * Nd, axis=1)) / OPT["wall_scale"])
        ** 2
    )
    return S, Ad, Nd, sw


def link_legs(legs, results):
    """Refine the fits of linked legs together (see the module docstring): every camera fitted on
    its own in two consecutive legs that share frames (reconstruct.py's leg overlap) gets one
    term holding the shared cameras together, LINK_SIGMA, weighted to count as much as the
    smaller leg's wall points. Returns ({leg id: {folder: params}} to place again, report)."""
    ents, report = {}, []
    for leg in legs:
        for folder, (entry, _, sh, fd) in results[leg["id"]].items():
            if fd is not None and entry.get("aligned"):
                ents[(leg["id"], folder)] = {
                    "fd": fd,
                    "x0": np.asarray(sh["fits"][0], float),
                }
    links = []
    for a, b in zip(legs, legs[1:]):
        if not a.get("link_next") or a["line"] != b["line"]:
            continue
        for folder in sorted(
            {k[1] for k in ents if k[0] == a["id"]}
            & {k[1] for k in ents if k[0] == b["id"]}
        ):
            fa, fb = ents[(a["id"], folder)]["fd"], ents[(b["id"], folder)]["fd"]
            common = sorted(set(fa["names"]) & set(fb["names"]))
            if len(common) >= 3:
                ia, ib = (
                    [fa["names"].index(n) for n in common],
                    [fb["names"].index(n) for n in common],
                )
                # the shared stretch must have the same shape in both legs (else one of them broke there)
                s_, ang, t = similarity_2d(fb["Lc"][ib], fa["Lc"][ia])
                Rm = np.array(
                    [[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]
                )
                m_per_unit = math.exp(ents[(a["id"], folder)]["x0"][1])
                shape = (
                    float(
                        np.median(
                            np.linalg.norm(
                                s_ * fb["Lc"][ib] @ Rm.T + t - fa["Lc"][ia], axis=1
                            )
                        )
                    )
                    * m_per_unit
                )
                if shape > LINK_SHAPE_M:
                    print(
                        f"link {a['id']}-{b['id']} {folder}: not linked, the shared cameras differ in shape by {shape:.1f} m"
                    )
                    report.append(
                        {
                            "legs": [a["id"], b["id"]],
                            "folder": folder,
                            "linked": False,
                            "reason": f"shared cameras differ in shape by {shape:.1f} m",
                        }
                    )
                    continue
                links.append(((a["id"], folder), (b["id"], folder), ia, ib, common))
    if not links:
        return {}, report
    while links:
        x0, x, keys, sl = solve_links(ents, links)
        worse = [
            k
            for k in keys
            if walls_median(ents, x, sl, k)
            > max(
                walls_median(ents, x0, sl, k) + LINK_WORSE_M,
                LINK_WORSE_X * walls_median(ents, x0, sl, k),
            )
        ]
        if not worse:
            break
        drop = [lk for lk in links if lk[0] in worse or lk[1] in worse]
        for lk in drop:
            print(
                f"link {lk[0][0]}-{lk[1][0]} {lk[0][1]}: undone, it made the wall fit of "
                + ", ".join(
                    f"{k[0]} {walls_median(ents, x0, sl, k):.2f} -> {walls_median(ents, x, sl, k):.2f} m"
                    for k in worse
                    if k in lk[:2]
                )
            )
            report.append(
                {
                    "legs": [lk[0][0], lk[1][0]],
                    "folder": lk[0][1],
                    "linked": False,
                    "reason": "made a leg's wall fit worse",
                }
            )
        links = [lk for lk in links if lk not in drop]
    if not links:
        return {}, report
    return finish_links(ents, links, x0, x, keys, sl, report)


def walls_median(ents, x, sl, k):
    fd = ents[k]["fd"]
    return float(
        np.median(
            np.minimum(
                segment_distances(apply_2d(x[sl[k]], fd["bpts"]), fd["A"], fd["B"]).min(
                    axis=1
                ),
                50,
            )
        )
    )


def solve_links(ents, links):
    keys = sorted({k for lk in links for k in lk[:2]})
    sl = {k: slice(4 * i, 4 * i + 4) for i, k in enumerate(keys)}
    x0 = np.concatenate([ents[k]["x0"] for k in keys])
    lower, upper = [], []
    for k in keys:
        a, ls, tx, ty = ents[k]["x0"]
        lower += [
            a - math.radians(5),
            ls - math.log(OPT["refine_scale"]),
            tx - 15,
            ty - 15,
        ]
        upper += [
            a + math.radians(5),
            ls + math.log(OPT["refine_scale"]),
            tx + 15,
            ty + 15,
        ]
    lower, upper = np.array(lower), np.array(upper)

    def link_gap(x, lk):
        ka, kb, ia, ib, _ = lk
        Pa = apply_2d(x[sl[ka]], ents[ka]["fd"]["Lc"][ia])
        Pb = apply_2d(x[sl[kb]], ents[kb]["fd"]["Lc"][ib])
        return Pa, Pb

    def path_len(P):
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    x = x0.copy()
    for _ in range(8):
        terms = {
            k: wall_terms(
                x[sl[k]], ents[k]["fd"]["bpts"], ents[k]["fd"]["A"], ents[k]["fd"]["B"]
            )
            for k in keys
        }
        wl = [
            math.sqrt(
                max(min(len(terms[lk[0]][0]), len(terms[lk[1]][0])), 1) / len(lk[4])
            )
            for lk in links
        ]

        def residuals(xx):
            r = []
            for k in keys:
                S, Ad, Nd, sw = terms[k]
                fd, prm, n = ents[k]["fd"], xx[sl[k]], max(len(S), 1)
                r.append(
                    sw
                    * np.sum((apply_2d(prm, S) - Ad) * Nd, axis=1)
                    / OPT["wall_scale"]
                )
                r.append(
                    math.sqrt(n)
                    * (apply_2d(prm, fd["Lc"]).mean(axis=0) - fd["gps"].mean(axis=0))
                    / OPT["gps_sigma"]
                )
                r.append(
                    [math.sqrt(n) * (prm[1] - fd["params0"][1]) / fd["scale_sigma"]]
                )
            for lk, w in zip(links, wl):
                Pa, Pb = link_gap(xx, lk)
                r.append(w * ((Pa - Pb) / LINK_SIGMA).ravel())
            return np.concatenate([np.ravel(v) for v in r])

        x = least_squares(
            residuals,
            np.clip(x, lower, upper),
            bounds=(lower, upper),
            x_scale=np.tile([0.01, 0.01, 1, 1], len(keys)),
        ).x
    return x0, x, keys, sl


def finish_links(ents, links, x0, x, keys, sl, report):
    def link_gap(xx, lk):
        ka, kb, ia, ib, _ = lk
        return apply_2d(xx[sl[ka]], ents[ka]["fd"]["Lc"][ia]), apply_2d(
            xx[sl[kb]], ents[kb]["fd"]["Lc"][ib]
        )

    def path_len(P):
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    for lk in links:
        g0, g1 = [
            np.linalg.norm(np.subtract(*link_gap(v, lk)), axis=1) for v in (x0, x)
        ]
        (a0, b0), (a1, b1) = link_gap(x0, lk), link_gap(x, lk)
        report.append(
            {
                "legs": [lk[0][0], lk[1][0]],
                "folder": lk[0][1],
                "linked": True,
                "shared_frames": len(lk[4]),
                "shared_cameras_apart_m": {
                    "before": float(np.median(g0)),
                    "after": float(np.median(g1)),
                },
                "scale_ratio": {
                    "before": path_len(a0) / max(path_len(b0), 1e-9),
                    "after": path_len(a1) / max(path_len(b1), 1e-9),
                },
            }
        )
        r = report[-1]
        print(
            f"link {lk[0][0]}-{lk[1][0]} {lk[0][1]}: {len(lk[4])} shared frames {r['shared_cameras_apart_m']['before']:.2f} -> "
            f"{r['shared_cameras_apart_m']['after']:.2f} m apart, scale ratio {r['scale_ratio']['before']:.3f} -> "
            f"{r['scale_ratio']['after']:.3f}"
        )
    for k in keys:
        print(
            f"  {k[0]}/{k[1]}: wall fit {walls_median(ents, x0, sl, k):.2f} -> {walls_median(ents, x, sl, k):.2f} m"
        )
    changed = {}
    for k in keys:
        changed.setdefault(k[0], {})[k[1]] = x[sl[k]]
    return changed, report


def align_roi(roi_dir, max_error=2.0, tag=None, link=True):
    t_start, secs = time.time(), {}
    p = RoiPaths(roi_dir, tag)
    if not p.legs_json.exists():
        raise SystemExit(f"{p.legs_json} not found: run reconstruct.py first")
    doc = json.loads(p.legs_json.read_text())
    p.alignment.mkdir(parents=True, exist_ok=True)
    # outputs of legs an earlier run planned but this one does not (e.g. rerun as one leg)
    current = {
        f"{leg['id']}_{cam['folder']}" for leg in doc["legs"] for cam in leg["cameras"]
    }
    for d in p.alignment.iterdir():
        if d.is_dir() and (d / "transform.json").exists() and d.name not in current:
            shutil.rmtree(d)
    for f in p.results.glob("fit_*.png") if p.results.exists() else []:
        if f.stem[len("fit_") :] not in current:
            f.unlink()
    roi = json.loads(p.roi_json.read_text())
    frames = {s["folder"]: p.frames(s["folder"]) for s in p.sequences()}
    lons = [f["lon"] for fr in frames.values() for f in fr.values()]
    lats = [f["lat"] for fr in frames.values() for f in fr.values()]
    ref = LocalFrame(*roi["lines"][0][0])
    pad = OPT["margin"] + OPT["search_m"] + 40
    # OSM outlines are downloaded once per ROI and reused: any earlier run's saved copy (tagged runs,
    # old results) is taken before asking Overpass, which is slow and often times out
    cache = p.alignment / "osm_buildings.json"
    saved = sorted(p.data.glob("alignment*/osm_buildings.json")) + sorted(
        p.data.glob("old/*/osm_buildings.json")
    )
    if not cache.exists() and saved:
        cache.write_bytes(saved[0].read_bytes())
    buildings = fetch_buildings(
        (
            min(lons) - pad / ref.kx,
            min(lats) - pad / ref.ky,
            max(lons) + pad / ref.kx,
            max(lats) + pad / ref.ky,
        ),
        cache=cache,
    )
    print(f"OSM: {len(buildings)} building outlines")
    secs["osm_buildings"] = time.time() - t_start

    def align_leg(leg, override=None):
        """Every camera of a leg: {folder: (entry, cameras_out, _shared, _fitdata)}. override: {folder:
        params} placing those cameras (link_legs' refined fits) instead of fitting them."""
        override = override or {}
        # a rig's reference camera first (its fit places the rig's other cameras); output in leg order
        refs = {c["rig"] for c in leg["cameras"] if c.get("rig") and c.get("model")}
        shared, out = {}, {}
        for cam in sorted(leg["cameras"], key=lambda c: c["folder"] not in refs):
            entry = {
                "leg": leg["id"],
                "folder": cam["folder"],
                "frames": cam["frames"],
                "cameras": [
                    {"folder": cam["folder"], "registered": cam.get("registered", 0)}
                ],
                "aligned": False,
            }
            if not cam.get("model"):
                entry["reason"] = "no reconstruction"
                out[cam["folder"]] = (entry, [], None, None)
                print(f"{leg['id']}/{cam['folder']}: no reconstruction")
                continue
            rig = cam.get("rig")
            out_dir = p.alignment / f"{leg['id']}_{cam['folder']}"
            prev = results.get(leg["id"], {}).get(cam["folder"])
            if cam["folder"] in override and prev and prev[2]:
                sh = {**prev[2], "fits": [override[cam["folder"]]]}
                res = align_camera(
                    p, leg, cam, buildings, out_dir, max_error, shared=sh
                )
                res["linked"] = True
            elif rig and rig == cam["folder"]:
                pool = [
                    c
                    for c in leg["cameras"]
                    if c.get("rig") == rig and c is not cam and c.get("model")
                ]
                res = align_camera(
                    p, leg, cam, buildings, out_dir, max_error, pool=pool, enu=ref
                )
            else:
                res = align_camera(
                    p,
                    leg,
                    cam,
                    buildings,
                    out_dir,
                    max_error,
                    shared=shared.get(rig) if rig else None,
                    enu=ref,
                )
            if rig and rig == cam["folder"] and res.get("aligned"):
                shared[rig] = res["_shared"]
            sh, fd = res.pop("_shared", None), res.pop("_fitdata", None)
            if prev and cam["folder"] in override:
                fd = prev[3]  # keep the first pass's fit data for the record
            cams_out = res.pop("cameras_out", [])
            entry.update(res)
            out[cam["folder"]] = (entry, cams_out, sh, fd)
            if res.get("aligned"):
                ptf = res["points_to_footprint_median_m"]
                print(
                    f"{leg['id']}/{cam['folder']}: {res['registered']}/{cam['frames']} cameras, "
                    f"{res['building_points']} building points: {ptf['mapillary_only']:.2f} -> {ptf['aligned']:.2f} m "
                    f"({100 * res['points_to_footprint_within_1m']:.0f}% within 1 m), scale x{res['scale_vs_mapillary']:.2f}, "
                    f"heading {res['heading_change_deg']:+.1f} deg, camera height {res['camera_height_m']:.2f} m, "
                    f"gravity spread {res['gravity']['spread_median_deg']:.1f} deg"
                    + (
                        f", {len(res['cameras_out_of_order'])} cameras behind earlier ones "
                        f"({', '.join(n[6:10] for n in res['cameras_out_of_order'])})"
                        if res["cameras_out_of_order"]
                        else ""
                    )
                    + (" (placed with the linked legs)" if res.get("linked") else "")
                    + f" -> {'reliable' if res['reliable'] else 'UNRELIABLE'}"
                )
            else:
                print(f"{leg['id']}/{cam['folder']}: not aligned ({res.get('reason')})")
        return out

    results = {}
    t0 = time.time()
    for leg in doc["legs"]:
        t1 = time.time()
        results[leg["id"]] = align_leg(leg)
        secs[f"fit {leg['id']}"] = time.time() - t1
    secs["fit_legs"] = time.time() - t0
    link_report = []
    if link and len(doc["legs"]) > 1:
        t0 = time.time()
        changed, link_report = link_legs(doc["legs"], results)
        secs["link_legs"] = time.time() - t0
        t0 = time.time()
        for leg in doc["legs"]:
            if changed.get(leg["id"]):
                print(f"{leg['id']}: placing again with the linked fit")
                results[leg["id"]] = align_leg(leg, changed[leg["id"]])
        secs["place_linked_legs"] = time.time() - t0

    legs_out, per_folder, n_frames = [], {}, 0
    for leg in doc["legs"]:
        for c in leg["cameras"]:
            entry, cams_out, _, _ = results[leg["id"]][c["folder"]]
            legs_out.append(entry)
            n_frames += c["frames"]
            # frames two legs share belong to the later leg (it starts with them)
            have = per_folder.setdefault(c["folder"], {})
            for co in cams_out:
                have[co["file"]] = co
    per_folder = {f: [d[k] for k in sorted(d)] for f, d in per_folder.items()}
    for folder, cams in per_folder.items():
        (p.alignment / f"{folder}.json").write_text(
            json.dumps({"folder": folder, "cameras": cams}, indent=1)
        )
    aligned = [e for e in legs_out if e.get("aligned")]
    summary = {
        "pipeline": "Alignment_Pipeline",
        "reconstruction": {
            k: doc.get(k)
            for k in (
                "features",
                "mapper",
                "refine_intrinsics",
                "gravity_priors",
                "mask_moving",
            )
        },
        "frames": n_frames,
        "leg_links": link_report,
        "frames_aligned": sum(e["registered"] for e in aligned),
        "frames_reliable": sum(e["registered"] for e in aligned if e["reliable"]),
        "legs": legs_out,
    }
    t0 = time.time()
    plot_overview(p, aligned, per_folder, buildings)
    from .plot_fit import plot_leg  # the sparse cloud on the footprints, plan and 3D

    for e in aligned:
        d = p.alignment / f"{e['leg']}_{e['folder']}"
        if (d / "transform.json").exists():
            plot_leg(p, d, buildings)
    secs["figures"] = time.time() - t0
    secs["total"] = time.time() - t_start
    summary["seconds"] = {k: round(v, 1) for k, v in secs.items()}
    p.summary_json.write_text(json.dumps(summary, indent=1))
    print(
        "align time: "
        + ", ".join(
            f"{k} {v:.0f} s" for k, v in secs.items() if not k.startswith("fit leg")
        )
    )
    print(
        f"{summary['frames_aligned']}/{n_frames} frames aligned, {summary['frames_reliable']} reliable; "
        f"wrote {p.summary_json}"
    )
    return summary


def plot_overview(p, aligned, per_folder, buildings):
    cams = [c for cs in per_folder.values() for c in cs]
    if not cams:
        return
    enu = LocalFrame(cams[0]["lon"], cams[0]["lat"])
    xy = np.array([enu.xy(c["lon"], c["lat"]) for c in cams])
    lo, hi = xy.min(axis=0) - 60, xy.max(axis=0) + 60
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, ring in buildings:
        r = np.array([enu.xy(lon, lat) for lon, lat in ring])
        if np.all(r.max(axis=0) >= lo) and np.all(r.min(axis=0) <= hi):
            ax.fill(*r.T, color="#e3e3e3", ec="#888", lw=0.5, zorder=0)
    for folder, cs in per_folder.items():
        a = np.array([enu.xy(c["lon"], c["lat"]) for c in cs])
        g = np.array([enu.xy(*c["mapillary_lonlat"]) for c in cs])
        rel = np.array([c["reliable"] for c in cs])
        ax.plot(*g.T, "x", color="#888", ms=3)
        ax.scatter(*a[rel].T, s=8, color="#1f4e9c", zorder=2)
        ax.scatter(*a[~rel].T, s=8, color="#d62728", zorder=2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(
        f"{p.name}: aligned cameras (blue reliable, red unreliable), Mapillary positions (x)",
        fontsize=10,
        loc="left",
    )
    p.results.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.results / "overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument(
        "--no-link",
        action="store_true",
        help="Fit each leg on its own (by default legs sharing frames are refined together)",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=2.0,
        help="Reliable if the median point-to-footprint distance is at most this (m)",
    )
    parser.add_argument("--tag", help="Separate run (see reconstruct.py --tag)")
    args = parser.parse_args()
    align_roi(args.roi_dir, args.max_error, args.tag, link=not args.no_link)


if __name__ == "__main__":
    main()
