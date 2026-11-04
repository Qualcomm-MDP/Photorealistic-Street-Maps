"""Stage 5 (reconstruct): camera poses and sparse points with ALIKED features and COLMAP 4.2.1's
global mapper (GLOMAP), using GeoCalib's calibration and gravity.

Legs: the primary sequence of each drawn line is split at turns sharper than --max-turn (its
frames heading within --trim-deg of the leg kept) and into chunks of at most --max-frames; each
rig camera (sibling sequence) gets its frames in the leg's time window. --max-turn 180
--trim-deg 180 --max-frames 200 keeps a whole route in one leg. Each side or rear camera of a leg
(looking more than 45 deg from travel) then keeps its longest stretch without a jump between
frames (Mapillary positions) of more than --gap-factor x its median spacing (and at least 6 m;
0: off): across such a gap it sees little or nothing in common, and a few false matches (e.g. on
road markings) are enough for the mapper to join the two sides wrongly. roi_11 leg01 (frames
8-39 m apart after a stop at a light): the rear camera's first 11 frames had been placed 55 m off
on 60 false matches of the centre lines; cutting them took the side cameras' bow from 19-40 m to
4-8 m. Forward cameras keep their frames: they still see down the street across such a jump
(roi_22 is reliable over 32-40 m jumps), and cutting roi_11's forward camera only lost buildings.

Per leg and sequence:
  1. Frames hard-linked into the work folder; feature masks for the burned-in overlay and for sky,
     vehicles and people from the segmentation.
  2. Camera (camera_setup): one per sequence from GeoCalib's focal, principal point at the
     centre; RADIAL_FISHEYE for strongly distorted lenses, RADIAL otherwise. A warning when
     Mapillary's focal differs by more than 10%. With --calibration mapillary: RADIAL from
     Mapillary's focal, k1 and k2 (calibrate.py --calibration mapillary first). Crops of a 360
     camera: their exact focal, no distortion, held fixed (nothing for bundle adjustment to refine).
  3. ALIKED keypoints, matched between each frame and its next 10, imported into COLMAP. Frames
     under 6 m apart (median, Mapillary positions): mutual nearest neighbour of the descriptors
     (nn_features.py, seconds per leg); further apart: LightGlue (~15x slower, but it holds the
     larger changes between neighbouring frames). --matcher nn|lightglue forces one.
  4. Gravity priors: GeoCalib's per-frame gravity in the database's pose_priors table (no
     position: GPS is not used here). COLMAP expects the gravity as the downward direction in the
     camera frame (-up_cam) and aligns the model's +y axis with it: checked on roi_2, where the
     cameras' world "down" came out (0, 1, 0) with the down vector, (0, -1, 0) with the up vector
     given instead, and in an arbitrary direction without priors.
  5. colmap global_mapper (GLOMAP) with gravity-aided rotation averaging (GlobalMapper.ra_use_gravity);
     bundle adjustment refines the focal and distortion from GeoCalib's start; then
     point_triangulator re-triangulates every match with the poses fixed and relaxed
     triangulation settings (more road and building points). Fallback: if any camera lies behind
     an earlier one along the leg (GLOMAP turned a few cameras wrongly and slid a stretch back),
     the leg is solved again with COLMAP's incremental mapper on the same matches, and that model
     is kept when it holds capture order with at least 90% of GLOMAP's frames
     (legs.json: mapper_used, mapper_fallback). --mapper incremental uses it from the start.
  6. Weak frames (only with --split-weak; off by default): frames left with under 0.35 x the
     leg's median number of 3D observations (e.g. a parked truck filling the view) are dropped
     and the leg solved again without them, and align.py fits each side of such a gap
     separately. It removed a 27 m spike on roi_11's truck leg, but on roi_20 the two sides of
     the gap ended 72 m apart where Mapillary has 41 m, so it is not on by default yet.
Writes data/reconstruction/legs.json and the models (data/reconstruction/<leg>/<folder>/sparse/<n>/txt).

Usage: python reconstruct.py rois/mapillary_roi_2 [--max-frames 60] [--tag t]
"""

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import (MASK_SKY_OR_MOVING, LocalFrame, RoiPaths, load_cameras, mapillary_calibration, model_size,
                    straight_stretches)

HERE = Path(__file__).resolve().parent
SENSOR_CAMERA = 0          # COLMAP 4 SensorType
COORDS_UNDEFINED = -1      # COLMAP 4 PosePrior::CoordinateSystem
RELAXED_TRI = ["--Mapper.tri_ignore_two_view_tracks", 0, "--Mapper.tri_min_angle", 1.0,
               "--Mapper.filter_min_tri_angle", 1.0]
# COLMAP's incremental mapper, thresholds relaxed for forward motion (it adds frames one at a time
# against the points already built, so a stretch cannot slide along the street the way GLOMAP's
# all-at-once positioning let roi_20's chunks slide 36-47 m back over each other)
INCREMENTAL = [*RELAXED_TRI, "--Mapper.abs_pose_min_num_inliers", 15, "--Mapper.abs_pose_min_inlier_ratio", 0.1,
               "--Mapper.init_min_tri_angle", 4, "--Mapper.init_max_forward_motion", 0.99]
MATCHERS = {"nn": "nearest-neighbour matching", "lightglue": "LightGlue"}
# COLMAP's mappers pick a random seed from the clock unless given one; a fixed seed makes a rerun give
# the same model, so two runs differ only by what was changed (roi_24: one leg's wall fit 0.62 m in one
# run, 0.84 m in the next with the same settings). --seed changes it
SEED = {"value": 0}
# Matching, per matcher: each frame with its next "overlap" frames ("look-ahead"); COLMAP's geometric
# verification (RANSAC) gives up on a pair after "ransac_trials" tries (COLMAP's default 10000).
# Frames close together (nearest-neighbour matching, under NN_MAX_SPACING m apart): 6 and 2000; 6 frames
# still reach ~18 m ahead at 3 m spacing. roi_13: 59 s and 29 s faster, same reliable frames, fits
# within ~0.2 m. Frames far apart (LightGlue) keep 10 and 10000: there the further neighbours hold the
# leg together (roi_19 with 6: all three legs 0.2-0.45 m worse; with 2000: two legs 0.2-0.3 m worse)
MATCH = {"nn": {"overlap": 6, "ransac_trials": 2000}, "lightglue": {"overlap": 10, "ransac_trials": 10000}}
SHARED_FOCAL_TOL = 0.01  # a leg camera whose own refined focal is this close to the shared one is not solved again
# Nearest-neighbour matching is ~15x faster than LightGlue and as good on frames up to ~5.6 m
# apart (roi_2, 7, 8, 11, main_street_east); on frames 8-10 m apart (roi_4, 5, 9) it finds too
# few matches between neighbours and the reconstruction becomes unstable (bends, lost frames)
NN_MAX_SPACING = 6.0  # metres between frames
WEAK_RATIO = 0.35          # frames under this x the leg's median 3D observations are dropped
CALIBRATION_NAMES = {"mapillary": "Mapillary", "pano": "the 360 crop (exact, fixed)"}
DISAGREE = 0.10            # warn when Mapillary's and GeoCalib's focals are further apart than this


# ----------------------------------------------------------------------------- legs

GAP_FLOOR_M = 6.0  # a jump between frames under this is never a gap
GAP_MIN_VIEW = 45.0  # only cameras looking further than this from travel (side, rear) are cut at gaps


def longest_contiguous(files, meta, factor, floor=GAP_FLOOR_M):
    """(the longest run of files with no step between neighbours over max(factor x the median step,
    floor) m, by Mapillary positions; the files left out). factor 0: all files.
    Distance alone, also for jumps between frames taken at the usual interval: those looked like GPS
    glitches, but Mapillary's own reconstruction jumped there too and the frames across did not
    reconstruct together (roi_13, keeping them: leg01 cam2 0.65 -> 3.02 m, cam1 scale end/start 4.95)."""
    if not factor or len(files) < 3:
        return files, []
    f0 = LocalFrame(meta[files[0]]["lon"], meta[files[0]]["lat"])
    xy = np.array([f0.xy(meta[n]["lon"], meta[n]["lat"]) for n in files])
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cuts = np.flatnonzero(steps > max(factor * np.median(steps), floor)) + 1
    bounds = [0, *cuts, len(files)]
    a, b = max(zip(bounds, bounds[1:]), key=lambda ab: ab[1] - ab[0])
    return files[a:b], files[:a] + files[b:]


LINK_MAX_GAP = 30  # frames between two legs (a turn) up to which the earlier one is extended into the next


def plan_legs(p, sequences, frames, settings):
    """Straight legs of each line's primary sequence, with every rig camera's frames in each
    leg's time window, each camera's cut to its longest stretch without a gap. With
    settings["leg_overlap"] = n, a leg runs on through the frames up to the next leg (a turn of
    up to LINK_MAX_GAP frames) and into its first n frames, so the two share those frames
    (leg["link_next"]: their primary-camera file names), which reconstruct_roi and align.py use
    to tie the legs together."""
    legs = []
    for line in sorted({s["line"] for s in sequences}):
        seqs = [s for s in sequences if s["line"] == line]
        primary = next(s for s in seqs if s["role"] == "primary")
        pf = sorted(frames[primary["folder"]].values(), key=lambda f: f["captured_at_ms"])
        frame = LocalFrame(pf[0]["lon"], pf[0]["lat"])
        chunks = []
        xy = [frame.xy(f["lon"], f["lat"]) for f in pf]
        for idx in straight_stretches(xy, [f["captured_at_ms"] for f in pf], settings["max_turn"], settings["trim_deg"]):
            core = [pf[i] for i in idx]
            if len(core) < settings["min_frames"]:
                continue
            n = math.ceil(len(core) / settings["max_frames"])
            chunks += [list(c) for c in np.array_split(np.array(core, dtype=object), n)]
        overlap, links = settings.get("leg_overlap", 0), [[] for _ in chunks]
        extras, core_end = [[] for _ in chunks], [None for _ in chunks]
        pos = {f["file"]: i for i, f in enumerate(pf)}
        for k in range(len(chunks) - 1 if overlap else 0):
            ia, ib = pos[chunks[k][-1]["file"]], pos[chunks[k + 1][0]["file"]]
            if ib - ia - 1 <= LINK_MAX_GAP:
                nxt = {f["file"] for f in chunks[k + 1]}
                extra = pf[ia + 1:max(ib, ia + 1) + overlap]
                links[k] = [f["file"] for f in extra if f["file"] in nxt]
                extras[k] = [f["file"] for f in extra]
                core_end[k] = chunks[k][-1]["captured_at_ms"]
                chunks[k] = chunks[k] + extra
        for core, link, extra, cend in zip(chunks, links, extras, core_end):
            leg_id = f"leg{len(legs) + 1:02d}"
            t0, t1 = core[0]["captured_at_ms"] - 500, core[-1]["captured_at_ms"] + 500
            pts = [frame.xy(f["lon"], f["lat"]) for f in core]
            cams = []
            for s in seqs:
                window = sorted(f["file"] for f in frames[s["folder"]].values() if t0 <= f["captured_at_ms"] <= t1)
                sideways = s["view_deg"] is not None and abs(s["view_deg"]) > GAP_MIN_VIEW
                window, cut = longest_contiguous(window, frames[s["folder"]], settings.get("gap_factor", 0) if sideways else 0)
                if len(window) >= 10:
                    cams.append({"folder": s["folder"], "sequence": s["sequence"], "role": s["role"],
                                 "view_deg": s["view_deg"], "frames": len(window), "frame_files": window,
                                 "gap_dropped": cut})
            leg = {"id": leg_id, "line": line, "primary_frames": len(core),
                   "length_m": round(sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))), "cameras": cams,
                   "link_next": link, "extra_frames": extra, "core_end_ms": cend}
            print(f"  {leg_id}: line {line}, {leg['length_m']} m{f', {len(link)} frames shared with the next leg' if link else ''}, "
                  + ", ".join(f"{c['folder']} {c['frames']} frames" + (f" ({len(c['gap_dropped'])} cut off by a gap)"
                                                                         if c["gap_dropped"] else "") for c in cams))
            legs.append(leg)
    if not legs:
        raise SystemExit("No straight stretch with enough frames; lower --min-frames or draw a longer route")
    return legs


# ----------------------------------------------------------------------------- camera

def camera_setup(p, folder, k1_limit=-0.2, focal=None):
    """(COLMAP camera model, params, description) for a sequence, from GeoCalib's median focal:
    RADIAL_FISHEYE (f, k1, k2 on the angle from the axis, starting undistorted) for a strongly
    distorted lens (GeoCalib k1 below k1_limit, e.g. wide action cameras), RADIAL (f, k1, k2,
    starting from GeoCalib's k1) otherwise. Bundle adjustment refines it. On roi_11's wide
    motorcycle camera the refined fisheye model kept the scale of a leg's second half at 0.93 of
    its first, RADIAL at 0.80-0.84.
    Mapillary's focal is recorded for comparison only: starting smartphones from it (roi_2, 4, 5,
    7, 8, 9) changed nothing where frames are close and did not steady widely spaced ones (roi_9
    collapsed), and it is sometimes far off (roi_15: 1.338).
    A Mapillary calibration (calibrate.py --calibration mapillary) starts RADIAL from Mapillary's
    focal, k1 and k2 instead: for lenses too wide for GeoCalib (roi_21's 2017 Roush rig: GeoCalib
    did not converge and gave its six identical cameras focals of 0.42-0.65, Mapillary 0.48)."""
    gc = json.loads(p.calibration_json(folder).read_text())
    w, h = gc["width"], gc["height"]
    make, mly = mapillary_calibration(p, folder)
    if gc.get("source") == "pano":  # a 360 crop: an exact pinhole (RADIAL with no distortion, so every reader takes it)
        params = [gc["focal_px"], w / 2, h / 2, 0.0, 0.0]
        return "RADIAL", params, {"source": "pano", "model": "RADIAL", "params": params, "make": make,
                                  "pano_focal_norm": gc["focal_norm"], "focal_disagreement": None}
    if gc.get("source") == "mapillary":
        params = [gc["focal_px"], w / 2, h / 2, gc["k1"], gc["k2"]]
        return "RADIAL", params, {"source": "mapillary", "model": "RADIAL", "params": params, "make": make,
                                  "mapillary_focal_norm": gc["focal_norm"], "focal_disagreement": None}
    fisheye = gc["k1"] < k1_limit
    model = "RADIAL_FISHEYE" if fisheye else "RADIAL"
    f_px = focal * max(w, h) if focal else gc["focal_px"]  # focal: fraction of the long side, overriding GeoCalib's
    params = [f_px, w / 2, h / 2, 0.0 if fisheye else gc["k1"], 0.0]
    info = {"source": "geocalib" if not focal else "geocalib, focal set to %.3f" % focal, "model": model, "params": params,
            "make": make,
            "geocalib_focal_norm": gc["focal_norm"], "geocalib_k1": gc["k1"],
            "mapillary_focal_norm": mly[0] if mly else None,
            "focal_disagreement": (gc["focal_norm"] and mly and mly[0] / gc["focal_norm"] - 1) or None}
    return model, params, info


# ----------------------------------------------------------------------------- COLMAP

def colmap(log, *args, check=True):
    """Run a COLMAP command, appending its output to the log file."""
    cmd = ["colmap", *map(str, args)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n" + result.stdout + result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"COLMAP {args[0]} failed ({result.returncode}); see {log}")
    return result


def run_python(log, script, *args):
    """Run one of this folder's scripts in its own Python process, appending its output to the log
    (PyTorch and COLMAP's libraries each ship an OpenMP runtime and are kept in separate processes)."""
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n" + result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed ({result.returncode}); see {log}")
    return result


def best_model(sparse):
    """(model with the most registered frames, sizes of COLMAP's pieces), or (None, [])."""
    models = [d for d in sparse.iterdir() if d.is_dir()] if sparse.exists() else []
    if not models:
        return None, []
    return max(models, key=model_size), sorted((model_size(d) for d in models), reverse=True)


def detect_overlay_rows(folder, sample=15, band=0.15):
    """Rows at the bottom of the frames covered by a static overlay, or 0.

    Overlay text and bonnets stay sharp and identical in every frame, while the scene
    behind them changes. A row is overlay if enough of its pixels are both static across
    frames and on an edge; the mask covers the bottom band from the highest such row."""
    paths = sorted(folder.glob("frame_*.jpg"))
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(sample, len(paths))).astype(int)]
    frames = [cv2.imread(str(q), cv2.IMREAD_GRAYSCALE) for q in paths]
    h, w = frames[0].shape
    scale = 480 / w
    small = np.stack([cv2.resize(f, (480, round(h * scale))).astype(np.float32) for f in frames])
    static = small.std(axis=0) < 6
    grad = np.hypot(*np.gradient(np.median(small, axis=0)))
    hits = (static & (grad > 25)).mean(axis=1)
    sh = small.shape[1]
    rows = np.flatnonzero(hits[int(sh * (1 - band)):] > 0.02)
    if not len(rows):
        return 0
    top = int(sh * (1 - band)) + rows.min()
    return int(np.ceil((sh - top) / scale)) + 10  # back to full resolution, plus a margin


def write_feature_masks(names, images, mask_dir, bottom_rows, seg_dir):
    """COLMAP feature masks (<image name>.png, black = ignore): the bottom overlay rows and the
    segmentation's sky/vehicle/person pixels. Returns mask_dir."""
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir(parents=True)
    for name in names:
        h, w = cv2.imread(str(images / name)).shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        if bottom_rows:
            mask[h - bottom_rows:] = 0
        seg = cv2.imread(str(seg_dir / f"{Path(name).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if seg is not None:
            mask[seg == MASK_SKY_OR_MOVING] = 0
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask)
    return mask_dir


# ----------------------------------------------------------------------------- reconstruction


def write_gravity_priors(database, gravity):
    """Gravity-only pose priors (COLMAP 4 pose_priors: one row per image, tied to it through its
    camera sensor; position NaN, so no position prior). gravity: {image name: 3-vector}."""
    con = sqlite3.connect(database)
    images = {name: (iid, cid) for name, iid, cid in con.execute("SELECT name, image_id, camera_id FROM images")}
    nan3, nan9 = np.full(3, np.nan).tobytes(), np.full(9, np.nan).tobytes()
    con.execute("DELETE FROM pose_priors")
    con.executemany("INSERT INTO pose_priors (corr_data_id, corr_sensor_id, corr_sensor_type, position, "
                    "position_covariance, gravity, coordinate_system) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(*images[n], SENSOR_CAMERA, nan3, nan9, np.asarray(g, np.float64).tobytes(), COORDS_UNDEFINED)
                     for n, g in gravity.items() if n in images])
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
    con.close()
    return n


WEAK_END_FRAMES = 10   # at most this many frames (rig frames) are left off either end of a leg ...
WEAK_END_RATIO = 0.1   # ... where the best link across the cut is under this x the leg's median best link


def weak_ends(database, groups, max_end=WEAK_END_FRAMES, ratio=WEAK_END_RATIO):
    """(image names to solve, image names left out, info): a leg's ends hanging on by few matches.
    groups: the leg's frames in capture order, each a list of image names (a rig frame's images
    together). The strength of a cut between two frames is its best link: the most verified inlier
    matches (COLMAP's two-view geometries) of any one image pair across it (a sum over the pairs would
    make every cut near an end look weak, fewer pairs crossing it). A cut within max_end frames of
    either end weaker than ratio x the median cut leaves the frames beyond it out. Such a tail is
    held by a few dozen matches against hundreds elsewhere, and the mapper places it with a wrong
    scale or behind the rest (roi_6 leg03 cam2: frames 0133-0136 held by 17-30 matches with the
    earlier frames, 255-678 between ordinary neighbours; the incremental mapper put them 45-49x too
    far apart, 80-165 m up the road)."""
    con = sqlite3.connect(database)
    ids = {name: iid for iid, name in con.execute("SELECT image_id, name FROM images")}
    group_of = {ids[n]: k for k, grp in enumerate(groups) for n in grp if n in ids}
    n = len(groups)
    strength = np.zeros(n)  # strength[k]: best link across the cut before group k
    for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
        i2 = pid % 2147483647
        a, b = group_of.get((pid - i2) // 2147483647), group_of.get(i2)
        if a is None or b is None or a == b:
            continue
        a, b = min(a, b), max(a, b)
        np.maximum(strength[a + 1:b + 1], rows, out=strength[a + 1:b + 1])  # the pair crosses every cut between a and b
    con.close()
    every = [x for grp in groups for x in grp]
    if n < 8:
        return every, [], {}
    ref = float(np.median(strength[1:]))
    weak = lambda k: strength[k] < ratio * ref  # noqa: E731
    lo = max([k for k in range(1, min(max_end, n - 1) + 1) if weak(k)], default=0)
    hi = min([k for k in range(max(1, n - max_end), n) if weak(k)], default=n)
    if (lo == 0 and hi == n) or hi - lo < max(5, n // 2):
        return every, [], {}
    keep = [x for grp in groups[lo:hi] for x in grp]
    info = {"median_best_link": round(ref), "front_frames_dropped": lo, "back_frames_dropped": n - hi,
            "front_cut_matches": int(strength[lo]) if lo else None, "back_cut_matches": int(strength[hi]) if hi < n else None}
    return keep, [x for x in every if x not in set(keep)], info


def observations_per_image(model_txt):
    """{image name: 2D points with a 3D point} from a COLMAP text model."""
    lines = [l for l in (Path(model_txt) / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    return {h.split()[9]: int((np.array(pts.split(), float).reshape(-1, 3)[:, 2] >= 0).sum()) if pts.strip() else 0
            for h, pts in zip(lines[::2], lines[1::2])}


def cameras_out_of_order(model_txt):
    """Frames placed behind an earlier one along the leg: a step against the direction from the first
    to the last camera of more than a third of a normal step (as align.py's reliability check)."""
    names, C, _ = load_cameras(model_txt)
    if len(names) < 3:
        return []
    d = C[-1] - C[0]
    steps = np.diff((C - C[0]) @ (d / max(np.linalg.norm(d), 1e-12)))
    return [names[i + 1] for i in np.flatnonzero(steps < -np.median(np.abs(steps)) / 3)]


def frame_spacing(p, folder, files):
    """Median distance (m) between consecutive frames, from Mapillary's positions."""
    meta = p.frames(folder)
    ll = np.array([(meta[n]["lon"], meta[n]["lat"]) for n in files])
    frame = LocalFrame(*ll[0])
    xy = np.array([frame.xy(*q) for q in ll])
    return float(np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0


def set_camera_params(database, params):
    """Overwrite the parameters of every camera in a COLMAP database (same model, new values)."""
    con = sqlite3.connect(database)
    con.execute("UPDATE cameras SET params = ?", (np.asarray(params, np.float64).tobytes(),))
    con.commit()
    con.close()


def count_priors(database):
    con = sqlite3.connect(database)
    n = con.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
    con.close()
    return n


def reconstruct_camera(p, leg, cam, overlays, log_prefix, matcher="auto", split_weak=False, mapper="global",
                       refine="all", focal=None, reuse=False, stage="all"):
    """stage: "all"; or "prepare" (frames and masks; returns the feature job for learned_features.py
    --jobs, reconstruct_roi running every leg camera's in one process) then "solve" (the rest).
    refine: what bundle adjustment refines of the camera: "all" (focal and distortion), "distortion"
    (focal held, e.g. on forward-only legs where the focal is weakly constrained and drifts, which
    stretches the model along the street) or "none". A 360 crop's exact camera is never refined.
    reuse: solve again on this leg camera's features and matches from an earlier run (the second,
    shared-focal pass of reconstruct_roi), with the camera reset to `focal`."""
    folder = cam["folder"]
    if p.pano(folder):
        refine, focal = "none", None
    refine_focal, refine_dist = refine == "all", refine in ("all", "distortion")
    secs, t_start = {}, time.time()

    def tick(key, t0):
        secs[key] = secs.get(key, 0.0) + time.time() - t0
    work = p.reconstruction / leg["id"] / folder
    images, sparse, log, db = work / "images", work / "sparse", work / "colmap.log", work / "database.db"
    reuse = reuse and db.exists()
    if reuse:
        for d in (sparse, work / "sparse_global"):
            shutil.rmtree(d, ignore_errors=True)
        sparse.mkdir()
        overlays.setdefault(folder, cam.get("mask_bottom", 0))
    elif stage == "solve":  # prepared, features extracted and matched
        masks = work / "feature_masks"
    else:
        for d in (images, sparse, work / "learned", work / "feature_masks"):
            if d.exists():
                shutil.rmtree(d)
        for f in (log, db):
            f.unlink(missing_ok=True)
        images.mkdir(parents=True)
        sparse.mkdir()
        for name in cam["frame_files"]:
            try:
                os.link(p.images / folder / name, images / name)  # COLMAP records symlinks by their targets
            except OSError:
                shutil.copy2(p.images / folder / name, images / name)
        if folder not in overlays:
            overlays[folder] = detect_overlay_rows(p.images / folder)
        masks = write_feature_masks(cam["frame_files"], images, work / "feature_masks", overlays[folder],
                                    p.masks / folder)

    gc = json.loads(p.calibration_json(folder).read_text())
    cmodel, cparams, cinfo = camera_setup(p, folder, focal=focal)
    if not reuse and stage != "solve" and cinfo.get("focal_disagreement") is not None and abs(cinfo["focal_disagreement"]) > DISAGREE:
        print(f"{log_prefix}: WARNING: GeoCalib's focal {cinfo['geocalib_focal_norm']:.3f} and Mapillary's "
              f"{cinfo['mapillary_focal_norm']:.3f} (of the long side) differ by "
              f"{100 * cinfo['focal_disagreement']:+.0f}%; starting from GeoCalib's", flush=True)
    spacing = frame_spacing(p, folder, cam["frame_files"])
    if matcher == "auto":
        matcher = "nn" if spacing < NN_MAX_SPACING else "lightglue"
    cam.update(matcher=matcher, frame_spacing_m=round(spacing, 2))
    if reuse:
        print(f"{log_prefix}: solving again with the focal held at {focal:.3f} of the long side "
              f"(the legs' shared focal), {mapper} mapper", flush=True)
        set_camera_params(db, cparams)
        n_priors = count_priors(db)
    else:
        if stage != "solve":
            print(f"{log_prefix}: {cam['frames']} frames {spacing:.1f} m apart, ALIKED + {MATCHERS[matcher]}, "
                  f"{cmodel} from {CALIBRATION_NAMES.get(cinfo['source'], 'GeoCalib')}, {mapper} mapper", flush=True)
        params = ",".join(f"{v:.8g}" for v in cparams)
        tick("prepare", t_start)
        if stage == "prepare":
            cam["_prepare_s"] = secs["prepare"]
            return {"images": str(images), "out": str(work / "learned"), "masks": str(masks),
                    "overlap": MATCH[matcher]["overlap"], "matcher": matcher}
        if stage == "solve":
            secs["prepare"] = cam.pop("_prepare_s", 0.0)
        t0 = time.time()
        if stage == "all":
            run_python(log, "learned_features.py", "--images", images, "--out", work / "learned",
                       "--overlap", MATCH[matcher]["overlap"], "--masks", masks, "--matcher", matcher)
        colmap(log, "feature_importer", "--database_path", db, "--image_path", images, "--import_path",
               work / "learned" / "features", "--ImageReader.single_camera", 1, "--ImageReader.camera_model",
               cmodel, "--ImageReader.camera_params", params)
        colmap(log, "matches_importer", "--database_path", db, "--match_list_path", work / "learned" / "matches.txt",
               "--match_type", "raw", "--FeatureMatching.use_gpu", 0, "--TwoViewGeometry.max_num_trials", MATCH[matcher]["ransac_trials"])
        gravity = {n: (-np.array(v["up_cam"])).tolist() for n, v in gc["frames"].items() if n in cam["frame_files"]}
        n_priors = write_gravity_priors(db, gravity)
        tick("features_matching", t0)

    def solve(image_list=None, use=mapper):
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir()
        t0 = time.time()
        if use == "global":
            colmap(log, "global_mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                   "--GlobalMapper.ra_use_gravity", 1, "--GlobalMapper.random_seed", SEED["value"], "--GlobalMapper.ba_refine_focal_length", int(refine_focal),
                   "--GlobalMapper.ba_refine_extra_params", int(refine_dist), "--GlobalMapper.ba_refine_principal_point", 0,
                   *(["--GlobalMapper.image_list_path", image_list] if image_list else []))
        else:
            colmap(log, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                   "--Mapper.ba_refine_focal_length", int(refine_focal), "--Mapper.ba_refine_extra_params", int(refine_dist),
                   *INCREMENTAL, "--Mapper.random_seed", SEED["value"],
                   *(["--Mapper.image_list_path", image_list] if image_list else []))
        tick("mapper" if use == mapper else "mapper_fallback", t0)
        t0 = time.time()
        for model in sorted(d for d in sparse.iterdir() if d.is_dir()):
            retri = sparse / f"_retri_{model.name}"
            retri.mkdir()
            colmap(log, "point_triangulator", "--database_path", db, "--image_path", images, "--input_path", model,
                   "--output_path", retri, "--clear_points", 1, "--refine_intrinsics", 0, *RELAXED_TRI)
            shutil.rmtree(model)
            retri.rename(model)
        for model in sorted(d for d in sparse.iterdir() if d.is_dir()):
            (model / "txt").mkdir()
            colmap(log, "model_converter", "--input_path", model, "--output_path", model / "txt", "--output_type", "TXT")
        tick("retriangulation", t0)
        return best_model(sparse)

    core, weak_out, weak_info = weak_ends(db, [[n] for n in cam["frame_files"]])
    image_list = None
    if weak_out:
        image_list = work / "image_list_core.txt"
        image_list.write_text("\n".join(core) + "\n")
        if not reuse:
            print(f"{log_prefix}: leaving out {len(weak_out)} end frames held by few matches "
                  f"({', '.join(n[6:10] for n in weak_out)}; {weak_info})", flush=True)
    cam.update(weak_end_dropped=weak_out, weak_end_info=weak_info)
    model, pieces = solve(image_list)
    dropped = {}
    if split_weak and model is not None and model_size(model) >= 3:
        # Frames the solve barely anchors (e.g. a parked truck filling the view: its matches are
        # rejected and it hides the buildings) get placed badly and can split the leg into parts at
        # different scales. Drop those with under WEAK_RATIO x the leg's median observation count
        # and solve the leg again without them (the matches across the gap keep it in one piece).
        counts = observations_per_image(model / "txt")
        med = float(np.median(list(counts.values())))
        dropped = {n: c for n, c in counts.items() if c < WEAK_RATIO * med}
        if dropped:
            keep = [n for n in core if n not in dropped]
            (work / "image_list.txt").write_text("\n".join(keep) + "\n")
            print(f"{log_prefix}: dropping {len(dropped)} weakly anchored frames "
                  + ", ".join(f"{n[6:10]} ({c} obs)" for n, c in sorted(dropped.items()))
                  + f" (leg median {med:.0f}) and solving again", flush=True)
            image_list = work / "image_list.txt"
            model, pieces = solve(image_list)
    used, fallback = mapper, None
    if mapper == "global" and model is not None and model_size(model) >= 3:
        # GLOMAP sets all rotations at once from the frame-pair rotations; where a few pairs agree on a
        # wrong rotation (a parked truck filling the view, a skipped frame) it turns those cameras by
        # degrees to tens of degrees, and placing them then slides a stretch back over the previous
        # one (roi_20, roi_11's truck leg; no seed or outlier threshold avoided it). The incremental
        # mapper places each frame against the points already built and keeps capture order, but
        # elsewhere GLOMAP is often better (roi_10, roi_5). So: GLOMAP, and the incremental mapper on
        # the same matches only where GLOMAP's cameras come out of order.
        behind = cameras_out_of_order(model / "txt")
        if behind:
            n_global = model_size(model)
            print(f"{log_prefix}: GLOMAP put {len(behind)} cameras behind earlier ones "
                  f"({', '.join(n[6:10] for n in behind)}); solving with the incremental mapper", flush=True)
            kept = work / "sparse_global"
            shutil.rmtree(kept, ignore_errors=True)
            sparse.rename(kept)
            model2, pieces2 = solve(image_list, "incremental")
            behind2 = cameras_out_of_order(model2 / "txt") if model2 is not None and model_size(model2) >= 3 else None
            fallback = {"global_out_of_order": behind, "global_registered": n_global,
                        "incremental_out_of_order": behind2,
                        "incremental_registered": model_size(model2) if model2 is not None else 0}
            if behind2 == [] and model_size(model2) >= 0.9 * n_global:
                used, model, pieces = "incremental", model2, pieces2
                shutil.rmtree(kept)
                print(f"{log_prefix}: incremental mapper keeps capture order ({model_size(model2)} frames); using it",
                      flush=True)
            else:
                shutil.rmtree(sparse)
                kept.rename(sparse)
                model, pieces = best_model(sparse)
                print(f"{log_prefix}: incremental mapper did not do better "
                      f"({fallback['incremental_registered']} frames, out of order: {behind2}); keeping GLOMAP's",
                      flush=True)
    cam.update(mapper_used=used, mapper_fallback=fallback,
               seconds={k: round(v, 1) for k, v in {**secs, "total": time.time() - t_start}.items()})
    cam["dropped_frames"] = {n: int(c) for n, c in sorted(dropped.items())}
    cam.update(calibration={**cinfo, "refined": refine}, gravity_priors=n_priors, colmap_pieces=pieces,
               mask_bottom=overlays[folder])
    if model is None or model_size(model) < 3:
        cam.update(model=None, registered=0)
        print(f"{log_prefix}: no model")
        return
    cam.update(model=str((model / "txt").relative_to(p.reconstruction)), registered=model_size(model))
    print(f"{log_prefix}: {cam['registered']}/{cam['frames']} registered with the {used} mapper"
          f"{f', {len(dropped)} weak frames dropped' if dropped else ''}"
          f"{f' (pieces {pieces})' if len(pieces) > 1 else ''}, {n_priors} gravity priors; "
          + ", ".join(f"{k} {v:.0f} s" for k, v in cam["seconds"].items()))


# ----------------------------------------------------------------------------- rigs

RIG_SYNC_MS = 20         # a rig camera's frame this close in time to a reference frame belongs to that rig frame
RIG_MIN_SYNC = 0.6       # cameras with fewer of their frames in sync with the reference are solved on their own:
# roi_6 leg03 at 32-33% (cameras firing on a 200 ms cycle, Mapillary keeping different frames of each) had
# too few shared rig frames to hold the side cameras, which came out turned (cam1 175 deg, cam2 32 deg;
# 7-8 m off the walls against 1.0-1.4 m solved alone); its legs at 65-100% and roi_11/13 (71-100%) are fine
RIG_CROSS_DEG = 60.0     # rig cameras looking within this of each other are also matched to each other
RIG_CROSS_FRAMES = 3     # ... each frame with the other camera's frames up to this many frame intervals away


def rig_members(p, leg):
    """(reference camera, [(camera, {frame file: image name in the joint work folder}, share in sync)])
    for the leg's rig: the primary is the reference; a sibling joins when at least RIG_MIN_SYNC
    of its frames were taken within RIG_SYNC_MS of a reference frame. Image names are
    <folder>/<time ms>.jpg, a frame in sync taking the reference frame's time, so COLMAP puts
    images with the same name (one per camera) in one rig frame."""
    cams = leg["cameras"]
    ref = next((c for c in cams if c["role"] == "primary"), cams[0])
    ref_meta = p.frames(ref["folder"])
    ref_t = np.array(sorted(ref_meta[f]["captured_at_ms"] for f in ref["frame_files"]))
    members = [(ref, {f: f"{ref['folder']}/{ref_meta[f]['captured_at_ms']}.jpg" for f in ref["frame_files"]}, 1.0)]
    for c in cams:
        if c is ref:
            continue
        meta = p.frames(c["folder"])
        names, used, synced = {}, set(), 0
        for f in c["frame_files"]:
            t = meta[f]["captured_at_ms"]
            j = int(np.argmin(np.abs(ref_t - t))) if len(ref_t) else -1
            if j >= 0 and abs(ref_t[j] - t) <= RIG_SYNC_MS and ref_t[j] not in used:
                key, synced = int(ref_t[j]), synced + 1
                used.add(key)
            else:
                key = t
            names[f] = f"{c['folder']}/{key}.jpg"
        share = synced / max(len(names), 1)
        if share >= RIG_MIN_SYNC:
            members.append((c, names, share))
    return ref, members


def mean_rotation(Rs):
    """Robust mean of rotation matrices (quaternion average, then again without those over 10 deg off)."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(np.asarray(Rs))
    m = q.mean()
    keep = np.degrees((q * m.inv()).magnitude()) <= 10
    return (q[keep].mean() if keep.sum() >= 3 else m).as_matrix()


def cam_from_ref_rotation(p, ref, cam):
    """A rig camera's rotation relative to the reference camera ((3, 3), how it was found):
    for 360 crops exactly from their yaws; otherwise the robust mean of Mapillary's computed
    rotations (its own reconstruction) at the frames both took at the same time; else from the
    compass headings alone (yaw only). Bundle adjustment refines it (not for 360 crops)."""
    from mapillary_layer import crop_axes
    pr, pc = p.pano(ref["folder"]), p.pano(cam["folder"])
    if pr and pc:
        return crop_axes(pc["yaw_deg"]).T @ crop_axes(pr["yaw_deg"]), "360 crop yaws (exact)"
    mr, mc = p.frames(ref["folder"]), p.frames(cam["folder"])
    by_t = {m["captured_at_ms"]: m for m in mr.values()}
    rel, yaw = [], []
    for m in mc.values():
        r = by_t.get(m["captured_at_ms"])
        if r is None:
            continue
        if r.get("computed_rotation") and m.get("computed_rotation"):
            rel.append(cv2.Rodrigues(np.array(m["computed_rotation"], float))[0]
                       @ cv2.Rodrigues(np.array(r["computed_rotation"], float))[0].T)
        if r.get("compass_angle") is not None and m.get("compass_angle") is not None:
            yaw.append(math.radians(m["compass_angle"] - r["compass_angle"]))
    if len(rel) >= 3:
        return mean_rotation(rel), f"Mapillary's rotations ({len(rel)} frames)"
    a = math.atan2(np.mean(np.sin(yaw)), np.mean(np.cos(yaw))) if yaw else 0.0
    # camera turned right by a about its y (down) axis: cam_from_ref = R_y(a)^T
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]), "compass headings (yaw only)"


def rot_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def rig_pairs(members, p, overlap=10):
    """Pairs to match: each camera's frames with its next `overlap` (time order), and cameras looking
    within RIG_CROSS_DEG of each other also across, frames up to RIG_CROSS_FRAMES intervals apart."""
    seqs, pairs = [], set()
    for cam, names, _ in members:
        meta = p.frames(cam["folder"])
        order = sorted(names, key=lambda f: meta[f]["captured_at_ms"])
        seqs.append((cam, [(meta[f]["captured_at_ms"], names[f]) for f in order]))
        for i, (_, a) in enumerate(seqs[-1][1]):
            pairs.update((a, b) for _, b in seqs[-1][1][i + 1:i + 1 + overlap])
    for i, (ca, sa) in enumerate(seqs):
        for cb, sb in seqs[i + 1:]:
            va, vb = ca.get("view_deg"), cb.get("view_deg")
            if va is None or vb is None or abs((va - vb + 180) % 360 - 180) > RIG_CROSS_DEG:
                continue
            dt = RIG_CROSS_FRAMES * float(np.median(np.diff([t for t, _ in sa]))) if len(sa) > 1 else 0
            tb = np.array([t for t, _ in sb])
            for t, a in sa:
                pairs.update((a, sb[j][1]) for j in np.flatnonzero(np.abs(tb - t) <= dt + RIG_SYNC_MS))
    return sorted(pairs)


def read_text_model(model_txt):
    """(camera lines by id, [(image header, points line)], {point id: (fields before track, [(image id, idx)])},
    rigs.txt data lines)."""
    model_txt = Path(model_txt)
    cams = {int(l.split()[0]): l for l in (model_txt / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")}
    lines = [l for l in (model_txt / "images.txt").read_text().splitlines() if not l.startswith("#")]
    images = list(zip(lines[0::2], lines[1::2]))
    points = {}
    for l in (model_txt / "points3D.txt").read_text().splitlines():
        if l and not l.startswith("#"):
            s = l.split()
            tr = list(map(int, s[8:]))
            points[int(s[0])] = (s[1:8], list(zip(tr[0::2], tr[1::2])))
    rigs = [l for l in (model_txt / "rigs.txt").read_text().splitlines() if l and not l.startswith("#")] \
        if (model_txt / "rigs.txt").exists() else []
    return cams, images, points, rigs


def split_rig_model(model_txt, members, out_root):
    """One ordinary single-camera text model per rig camera (images renamed back to frame_XXXX.jpg,
    the points with at least two of that camera's observations), in the joint model's world frame,
    so align.py and later stages read it like any leg camera's model. Returns {folder: (model dir,
    registered)}."""
    cams, images, points, _ = read_text_model(model_txt)
    out = {}
    for cam, names, _ in members:
        back = {v: k for k, v in names.items()}
        kept = [(h, o) for h, o in images if h.split()[9] in back]
        if not kept:
            out[cam["folder"]] = (None, 0)
            continue
        ids = {int(h.split()[0]) for h, _ in kept}
        cam_id = int(kept[0][0].split()[8])
        pts = {}
        for pid, (head, track) in points.items():
            tr = [(i, k) for i, k in track if i in ids]
            if len(tr) >= 2:
                pts[pid] = (head, tr)
        d = Path(out_root) / cam["folder"] / "txt"
        d.mkdir(parents=True, exist_ok=True)
        (d / "cameras.txt").write_text("# Camera list\n" + cams[cam_id] + "\n")
        img_lines, frame_lines = [], []
        for h, o in sorted(kept, key=lambda x: back[x[0].split()[9]]):
            s = h.split()
            v = o.split()
            for k in range(2, len(v), 3):
                if int(v[k]) != -1 and int(v[k]) not in pts:
                    v[k] = "-1"
            img_lines += [" ".join(s[:9] + [back[s[9]]]), " ".join(v)]
            frame_lines.append(f"{s[0]} 1 {' '.join(s[1:8])} 1 CAMERA {cam_id} {s[0]}")
        (d / "images.txt").write_text("# Image list\n" + "\n".join(img_lines) + "\n")
        (d / "points3D.txt").write_text("# 3D point list\n" + "\n".join(
            " ".join([str(pid), *head, *(f"{i} {k}" for i, k in tr)]) for pid, (head, tr) in pts.items()) + "\n")
        (d / "rigs.txt").write_text(f"# Rig list\n1 1 CAMERA {cam_id}\n")
        (d / "frames.txt").write_text("# Frame list\n" + "\n".join(frame_lines) + "\n")
        out[cam["folder"]] = (d, len(kept))
    return out


def solved_cam_from_rig(model_txt):
    """{camera id: (3, 3) rotation} of the joint model's rig (rigs.txt)."""
    from common import quat_to_rot
    _, _, _, rigs = read_text_model(model_txt)
    out = {}
    for l in rigs:
        s = l.split()
        k = 4
        out[int(s[3])] = np.eye(3)
        while k < len(s):
            sid, has = int(s[k + 1]), int(s[k + 2])
            if has:
                out[sid] = quat_to_rot(*map(float, s[k + 3:k + 7]))
                k += 10
            else:
                k += 3
    return out


def reconstruct_rig(p, leg, members, ref, overlays, matcher="auto", mapper="global", refine="all", focal=None,
                    focals=None, reuse=False, stage="all"):
    """All of a leg's rig cameras in one COLMAP reconstruction with rig constraints: each rig frame
    (the cameras' images taken at the same moment) has one pose, and each camera a fixed pose in
    the rig, so the forward and rear cameras, which hold their direction well, keep the side
    cameras from bending (roi_11: side cameras solved alone bowed 2-40 m off a straight street).
    The rig is solved twice. First with the cameras' rotations in the rig starting from
    cam_from_ref_rotation and refined, translations from zero: the rotations come out well (each
    camera's own frames fix them), but a translation is not observable when no point is seen by
    that camera and another (a side camera matches only itself), so bundle adjustment slides the
    camera with its points (roi_11: 2-12 m off the reference, facades 2-5 m off between cameras).
    Then again with those rotations held and every camera at the reference camera's position (the
    real offsets, well under a metre, are far smaller than that drift). 360 crops: the rig is exact
    and solved once, held. Otherwise as reconstruct_camera: the
    same features, masks, calibration (camera_setup per camera), gravity priors, mapper and
    incremental fallback; then the joint model is split into one model per camera
    (split_rig_model), so the later stages run per camera as before. focals: {folder: focal} per
    camera (the legs' shared focal), overriding `focal`. reuse: solve again on an earlier run's
    features and matches (reconstruct_camera). stage: as reconstruct_camera's."""
    t_start, secs = time.time(), {}

    def tick(key, t0):
        secs[key] = secs.get(key, 0.0) + time.time() - t0
    work = p.reconstruction / leg["id"] / "rig"
    images, sparse, log, db = work / "images", work / "sparse", work / "colmap.log", work / "database.db"
    reuse = reuse and db.exists()
    keep_files = reuse or stage == "solve"
    for d in ((sparse, work / "sparse_global", work / "split") if keep_files else
              (images, sparse, work / "learned", work / "feature_masks", work / "split")):
        if d.exists():
            shutil.rmtree(d)
    if not keep_files:
        for f in (log, db):
            f.unlink(missing_ok=True)
    masks = work / "feature_masks"
    pano = all(p.pano(c["folder"]) for c, _, _ in members)
    if pano:
        refine = "none"
    refine_focal, refine_dist = refine == "all", refine in ("all", "distortion")
    config, infos, gravity = [], {}, {}
    for cam, names, share in members:
        folder = cam["folder"]
        if keep_files:
            overlays.setdefault(folder, cam.get("mask_bottom", 0))
        else:
            (images / folder).mkdir(parents=True)
            (masks / folder).mkdir(parents=True)
            if folder not in overlays:
                overlays[folder] = detect_overlay_rows(p.images / folder)
        for f, n in ({} if keep_files else names).items():
            try:
                os.link(p.images / folder / f, images / n)
            except OSError:
                shutil.copy2(p.images / folder / f, images / n)
            h, w = cv2.imread(str(images / n)).shape[:2]
            m = np.full((h, w), 255, np.uint8)
            if overlays[folder]:
                m[h - overlays[folder]:] = 0
            seg = cv2.imread(str(p.masks / folder / f"{Path(f).stem}.png"), cv2.IMREAD_GRAYSCALE)
            if seg is not None:
                m[seg == MASK_SKY_OR_MOVING] = 0
            cv2.imwrite(str(masks / f"{n}.png"), m)
        cmodel, cparams, cinfo = camera_setup(p, folder, focal=None if p.pano(folder) else
                                              (focals or {}).get(folder, focal))
        infos[folder] = cinfo
        entry = {"image_prefix": f"{folder}/", "camera_model_name": cmodel,
                 "camera_params": [float(v) for v in cparams]}
        if cam is ref:
            entry["ref_sensor"] = True
            cam["rig_start"] = {"from": "reference camera"}
        else:
            R, how = cam_from_ref_rotation(p, ref, cam)
            entry["cam_from_rig_rotation"] = rot_to_quat_wxyz(R)
            entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
            cam["rig_start"] = {"from": how, "cam_from_ref_quat_wxyz": entry["cam_from_rig_rotation"]}
        cam["rig_sync_share"] = round(share, 3)
        config.append(entry)
        gc = json.loads(p.calibration_json(folder).read_text())
        gravity.update({names[f]: (-np.array(v["up_cam"])).tolist() for f, v in gc["frames"].items() if f in names})
    (work / "rig_config.json").write_text(json.dumps([{"cameras": config}], indent=1))
    spacing = frame_spacing(p, ref["folder"], ref["frame_files"])
    if matcher == "auto":
        matcher = "nn" if spacing < NN_MAX_SPACING else "lightglue"
    pairs = rig_pairs(members, p, MATCH[matcher]["overlap"])
    (work / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in pairs))
    label = f"{leg['id']}/rig"
    if stage != "solve":
        print(f"{label}: {len(members)} cameras ({', '.join(c['folder'] for c, _, _ in members)}), "
              f"{sum(len(n) for _, n, _ in members)} images in {len({Path(x).name for _, n, _ in members for x in n.values()})} rig frames, "
              f"{len(pairs)} pairs, ALIKED + {MATCHERS[matcher]}, {mapper} mapper", flush=True)
    tick("prepare", t_start)
    if stage == "prepare":
        ref["_prepare_s"] = secs["prepare"]
        return {"images": str(images), "out": str(work / "learned"), "masks": str(masks),
                "pairs": str(work / "pairs.txt"), "matcher": matcher}
    if stage == "solve":
        secs["prepare"] = ref.pop("_prepare_s", 0.0)
    t0 = time.time()
    if reuse:  # the cameras reset to this pass's calibration by the rig configuration
        colmap(log, "rig_configurator", "--database_path", db, "--rig_config_path", work / "rig_config.json")
        n_priors = count_priors(db)
    else:
        if stage == "all":
            run_python(log, "learned_features.py", "--images", images, "--out", work / "learned", "--masks", masks,
                       "--pairs", work / "pairs.txt", "--matcher", matcher)
        colmap(log, "feature_importer", "--database_path", db, "--image_path", images, "--import_path",
               work / "learned" / "features", "--ImageReader.single_camera_per_folder", 1)
        colmap(log, "rig_configurator", "--database_path", db, "--rig_config_path", work / "rig_config.json")
        colmap(log, "matches_importer", "--database_path", db, "--match_list_path", work / "learned" / "matches.txt",
               "--match_type", "raw", "--FeatureMatching.use_gpu", 0, "--TwoViewGeometry.max_num_trials", MATCH[matcher]["ransac_trials"])
        n_priors = write_gravity_priors(db, gravity)
    tick("features_matching", t0)

    folder_of_prefix = {e["image_prefix"].rstrip("/"): e for e in config}
    first_rot = {}  # folder: the first solve's rotation in the rig

    # rig frames in capture order (images named <folder>/<time ms>.jpg); weakly held ends left out
    by_time = {}
    for _, names, _ in members:
        for x in names.values():
            by_time.setdefault(int(Path(x).stem), []).append(x)
    core, weak_out, weak_info = weak_ends(db, [by_time[k] for k in sorted(by_time)])
    rig_list = None
    if weak_out:
        rig_list = work / "image_list_core.txt"
        rig_list.write_text("\n".join(core) + "\n")
        if not reuse:
            print(f"{label}: leaving out {len({Path(x).stem for x in weak_out})} end rig frames held by few matches "
                  f"({weak_info})", flush=True)
    for cam, names, _ in members:
        back = {v: k for k, v in names.items()}
        cam.update(weak_end_dropped=sorted(back[x] for x in weak_out if x in back), weak_end_info=weak_info)

    def run_mapper(use, refine_rig):
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir()
        if use == "global":
            colmap(log, "global_mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                   "--GlobalMapper.ra_use_gravity", 1, "--GlobalMapper.random_seed", SEED["value"], "--GlobalMapper.ba_refine_focal_length", int(refine_focal),
                   "--GlobalMapper.ba_refine_extra_params", int(refine_dist), "--GlobalMapper.ba_refine_principal_point", 0,
                   "--GlobalMapper.refine_sensor_from_rig", int(refine_rig),
                   *(["--GlobalMapper.image_list_path", rig_list] if rig_list else []))
        else:
            colmap(log, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                   "--Mapper.ba_refine_focal_length", int(refine_focal), "--Mapper.ba_refine_extra_params", int(refine_dist),
                   "--Mapper.ba_refine_sensor_from_rig", int(refine_rig), *INCREMENTAL,
                   "--Mapper.random_seed", SEED["value"], *(["--Mapper.image_list_path", rig_list] if rig_list else []))

    def solve(use):
        t0 = time.time()
        run_mapper(use, not pano)
        if not pano:
            # second solve: the first solve's rotations held, every camera at the reference's position
            first = max((d for d in sparse.iterdir() if d.is_dir()), key=lambda d: (d / "images.bin").stat().st_size
                        if (d / "images.bin").exists() else 0, default=None)
            if first is not None:
                (first / "txt").mkdir(exist_ok=True)
                colmap(log, "model_converter", "--input_path", first, "--output_path", first / "txt", "--output_type", "TXT")
                _, imgs, _, _ = read_text_model(first / "txt")
                prefix_of = {int(h.split()[8]): h.split()[9].split("/")[0] for h, _ in imgs}
                for cid, R in solved_cam_from_rig(first / "txt").items():
                    e = folder_of_prefix.get(prefix_of.get(cid))
                    if e is not None and not e.get("ref_sensor"):
                        e["cam_from_rig_rotation"] = rot_to_quat_wxyz(R)
                        e["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
                        first_rot[prefix_of[cid]] = R
                (work / "rig_config_held.json").write_text(json.dumps([{"cameras": config}], indent=1))
                colmap(log, "rig_configurator", "--database_path", db, "--rig_config_path", work / "rig_config_held.json")
                run_mapper(use, False)
        tick("mapper" if use == mapper else "mapper_fallback", t0)
        t0 = time.time()
        for model in sorted(d for d in sparse.iterdir() if d.is_dir()):
            retri = sparse / f"_retri_{model.name}"
            retri.mkdir()
            colmap(log, "point_triangulator", "--database_path", db, "--image_path", images, "--input_path", model,
                   "--output_path", retri, "--clear_points", 1, "--refine_intrinsics", 0, *RELAXED_TRI)
            shutil.rmtree(model)
            retri.rename(model)
            (model / "txt").mkdir()
            colmap(log, "model_converter", "--input_path", model, "--output_path", model / "txt", "--output_type", "TXT")
        tick("retriangulation", t0)
        model, pieces = best_model(sparse)
        if model is None:
            return None, pieces, {}
        split = work / "split" / use
        shutil.rmtree(split, ignore_errors=True)
        return model, pieces, split_rig_model(model / "txt", members, split)

    def order_bad(split):
        d, n = split.get(ref["folder"], (None, 0))
        return cameras_out_of_order(d) if d is not None and n >= 3 else None

    model, pieces, split = solve(mapper)
    used, fallback = mapper, None
    behind = order_bad(split) if model is not None else None
    if mapper == "global" and behind:
        n_global = model_size(model)
        print(f"{label}: GLOMAP put {len(behind)} reference frames behind earlier ones; solving with the incremental mapper",
              flush=True)
        kept = work / "sparse_global"
        shutil.rmtree(kept, ignore_errors=True)
        sparse.rename(kept)
        model2, pieces2, split2 = solve("incremental")
        behind2 = order_bad(split2) if model2 is not None else None
        fallback = {"global_out_of_order": behind, "global_registered": n_global, "incremental_out_of_order": behind2,
                    "incremental_registered": model_size(model2) if model2 is not None else 0}
        if behind2 == [] and model_size(model2) >= 0.9 * n_global:
            used, model, pieces, split = "incremental", model2, pieces2, split2
            shutil.rmtree(kept)
        else:
            shutil.rmtree(sparse)
            kept.rename(sparse)
            model, pieces = best_model(sparse)
            split = split_rig_model(model / "txt", members, work / "split" / "global")
        print(f"{label}: using the {used} mapper", flush=True)
    solved = solved_cam_from_rig(model / "txt") if model is not None else {}
    seconds = {k: round(v, 1) for k, v in {**secs, "total": time.time() - t_start}.items()}
    for cam, names, _ in members:
        folder = cam["folder"]
        d, n = split.get(folder, (None, 0))
        cam.update(rig=ref["folder"], matcher=matcher, frame_spacing_m=round(frame_spacing(p, folder, cam["frame_files"]), 2),
                   mapper_used=used, mapper_fallback=fallback, dropped_frames={},
                   seconds=seconds if cam is ref else {}, calibration={**infos[folder], "refined": refine},
                   gravity_priors=sum(1 for x in names.values() if x in gravity), colmap_pieces=pieces,
                   mask_bottom=overlays[folder])
        if d is not None and n >= 3:
            cam_id = int(next(l for l in (d / "cameras.txt").read_text().splitlines() if not l.startswith("#")).split()[0])
            if cam is not ref and cam_id in solved:
                R0 = np.array(cam["rig_start"]["cam_from_ref_quat_wxyz"])
                from common import quat_to_rot
                diff = solved[cam_id] @ quat_to_rot(*R0).T
                cam["rig_solved_change_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(diff) - 1) / 2, -1, 1)))), 2)
            cam.update(model=str(d.relative_to(p.reconstruction)), registered=n)
        else:
            cam.update(model=None, registered=0)
        print(f"{label}/{folder}: {cam['registered']}/{cam['frames']} registered"
              + (f", rig rotation moved {cam['rig_solved_change_deg']:.1f} deg from {cam['rig_start']['from']}"
                 if "rig_solved_change_deg" in cam else ""), flush=True)
    print(f"{label}: {n_priors} gravity priors, pieces {pieces}; "
          + ", ".join(f"{k} {v:.0f} s" for k, v in seconds.items()), flush=True)


def overlap_ok(p, leg, frames, max_ratio=1.5):
    """(ok, why): whether a leg's frames run on into the next leg reconstructed like the rest of it:
    its primary camera's cameras in capture order, and their spacing against Mapillary's (the
    model's step over the GPS step, median) within max_ratio of the rest of the leg's."""
    extra = set(leg.get("extra_frames") or [])
    cam = next((c for c in leg["cameras"] if c["role"] == "primary"), None)
    if not extra or cam is None or not cam.get("model"):
        return True, ""
    model = p.reconstruction / cam["model"]
    behind = cameras_out_of_order(model)
    if behind:
        return False, f"{len(behind)} cameras behind earlier ones"
    names, C, _ = load_cameras(model)
    meta = frames[cam["folder"]]
    f0 = LocalFrame(meta[names[0]]["lon"], meta[names[0]]["lat"])
    g = np.array([f0.xy(meta[n]["lon"], meta[n]["lat"]) for n in names])
    r = np.linalg.norm(np.diff(C, axis=0), axis=1) / np.maximum(np.linalg.norm(np.diff(g, axis=0), axis=1), 0.5)
    in_extra = np.array([names[i + 1] in extra for i in range(len(names) - 1)])
    if in_extra.sum() < 2 or (~in_extra).sum() < 2:
        return True, ""
    q = float(np.median(r[in_extra]) / np.median(r[~in_extra]))
    if not 1 / max_ratio <= q <= max_ratio:
        return False, f"their spacing {q:.2f}x the rest of the leg's against GPS"
    return True, ""


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])


def model_focal(model_txt):
    """(focal as a fraction of the long side, image width, height) of a single-camera text model."""
    line = next(l for l in (Path(model_txt) / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")).split()
    return float(line[4]) / max(int(line[2]), int(line[3])), int(line[2]), int(line[3])


def reconstruct_roi(roi_dir, max_turn=35.0, trim_deg=20.0, min_frames=15, max_frames=60, tag=None, matcher="auto",
                    split_weak=False, mapper="global", calibration="geocalib", refine="all", focal=None, gap_factor=3.0,
                    rig=True, leg_overlap=8, share_focal=True):
    """Every leg, a leg's rig cameras together (siblings taking frames at the same moments as the
    primary; rig=False, --solo-cameras: each camera on its own, except 360 crops, always a rig). With
    more than one leg and share_focal, a second pass: each camera's focal over the legs (the
    median of the legs' refined focals, weighted by their frames) and every leg solved again on
    its own features and matches with the focal held there and the distortion refined. One
    camera has one focal; solved per leg, the focal of a forward-looking leg is weakly held and
    drifts, which stretches that leg along the street (roi_27: 0.84 -> fit 0.89 m, 0.81 -> 0.59 m)."""
    t_start, run_secs = time.time(), {}
    p = RoiPaths(roi_dir, tag, calibration)
    sequences = p.sequences()
    frames = {s["folder"]: p.frames(s["folder"]) for s in sequences}
    for s in sequences:
        if not p.calibration_json(s["folder"]).exists():
            raise SystemExit(f"No {calibration} calibration for {s['folder']}: run the calibrate stage first"
                             f"{' with --calibration mapillary' if calibration == 'mapillary' else ''}")
    settings = {"max_turn": max_turn, "trim_deg": trim_deg, "min_frames": min_frames, "max_frames": max_frames,
                "gap_factor": gap_factor, "leg_overlap": leg_overlap}
    p.reconstruction.mkdir(parents=True, exist_ok=True)
    print("Planning legs:")
    legs = plan_legs(p, sequences, frames, settings)
    doc = {"pipeline": "Alignment_Pipeline", "settings": settings, "calibration": calibration, "features": "aliked",
           "mapper": mapper,
           "refine_intrinsics": refine, "focal_override": focal, "mask_moving": True, "gravity_priors": True, "split_weak": split_weak, "weak_frame_ratio": WEAK_RATIO,
           "matcher": matcher, "rig": rig, "shared_focal": None, "seed": SEED["value"], "match": json.loads(json.dumps(MATCH)), "tag": tag,
           "legs": legs}
    overlays = {}

    def close_to_shared(cam, focals):
        """Its first-pass focal already within SHARED_FOCAL_TOL of the shared one: no second solve."""
        f, f1 = (focals or {}).get(cam["folder"]), cam.get("focal_first_pass")
        return f is not None and f1 is not None and abs(f1 / f - 1) <= SHARED_FOCAL_TOL

    def solve_leg(leg, refine, focals=None, reuse=False, stage="all", jobs=None):
        in_rig = set()
        if (rig or all(p.pano(c["folder"]) for c in leg["cameras"])) and len(leg["cameras"]) > 1:
            ref, members = rig_members(p, leg)
            if len(members) > 1:
                in_rig = {c["folder"] for c, _, _ in members}
                if reuse and all(close_to_shared(c, focals) for c, _, _ in members):
                    for c, _, _ in members:
                        c["shared_focal_skipped"] = True
                    print(f"{leg['id']}/rig: every camera's focal within {100 * SHARED_FOCAL_TOL:g}% of the shared one; not solved again")
                    members = []
                first = {c["folder"]: dict(c.get("seconds") or {}) for c, _, _ in members}
                try:
                    if members:
                        job = reconstruct_rig(p, leg, members, ref, overlays, matcher, mapper, refine, focal, focals,
                                              reuse, stage)
                        if stage == "prepare":
                            jobs.append(job)
                except RuntimeError as e:
                    for c, _, _ in members:
                        c.update(model=None, registered=0)
                    print(f"{leg['id']}/rig: failed: {e}")
                if reuse:
                    for c, _, _ in members:
                        c["shared_focal_skipped"] = False
                        c["seconds"] = {**first[c["folder"]], **({"shared_focal_pass": c["seconds"]["total"]}
                                                                  if c.get("seconds") else {})}
                p.legs_json.write_text(json.dumps(doc, indent=1))
        for cam in leg["cameras"]:
            if cam["folder"] in in_rig:
                continue
            label = f"{leg['id']}/{cam['folder']}"
            first = dict(cam.get("seconds") or {})
            f = (focals or {}).get(cam["folder"], focal)
            if reuse and f is None:
                continue
            if reuse:
                cam["shared_focal_skipped"] = close_to_shared(cam, focals)
                if cam["shared_focal_skipped"]:
                    print(f"{label}: focal {cam['focal_first_pass']:.3f} within {100 * SHARED_FOCAL_TOL:g}% of the shared "
                          f"{f:.3f}; not solved again")
                    continue
            try:
                job = reconstruct_camera(p, leg, cam, overlays, label, matcher, split_weak, mapper, refine, f, reuse, stage)
                if stage == "prepare":
                    jobs.append(job)
            except RuntimeError as e:
                cam.update(model=None, registered=0)
                print(f"{label}: failed: {e}")
            if reuse:
                cam["seconds"] = {**first, "shared_focal_pass": cam.get("seconds", {}).get("total", 0.0)}
            p.legs_json.write_text(json.dumps(doc, indent=1))

    run_secs["plan_legs"] = time.time() - t_start
    # Features for every leg camera in one process: each frame's keypoints extracted once (legs overlap,
    # and a rig leg and the shared-focal pass reuse them), the networks loaded once
    t0 = time.time()
    jobs = []
    for leg in legs:
        solve_leg(leg, refine, stage="prepare", jobs=jobs)
    run_secs["prepare"] = time.time() - t0
    t0 = time.time()
    jobs_file = p.reconstruction / "feature_jobs.json"
    jobs_file.write_text(json.dumps(jobs, indent=1))
    out = run_python(p.reconstruction / "features.log", "learned_features.py", "--jobs", jobs_file)
    batch = json.loads(out.stdout.strip().splitlines()[-1])
    run_secs["features_and_matching"] = time.time() - t0
    doc["features_batch"] = batch
    print(f"Features: {len(jobs)} leg cameras in one process, {batch['images_extracted']} frames extracted "
          f"({batch['images_reused']} reused), {run_secs['features_and_matching']:.0f} s", flush=True)
    t0 = time.time()
    for leg in legs:
        t1 = time.time()
        solve_leg(leg, refine, stage="solve")
        ok, why = overlap_ok(p, leg, frames)
        if not ok:
            # the frames run on into the next leg (often round a bend) broke this leg: solve it without them
            print(f"{leg['id']}: the frames shared with the next leg did not reconstruct well ({why}); "
                  "solving the leg without them, not linked to the next", flush=True)
            t_end = leg["core_end_ms"] + 500
            for cam in leg["cameras"]:
                meta = frames[cam["folder"]]
                cam["frame_files"] = [f for f in cam["frame_files"] if meta[f]["captured_at_ms"] <= t_end]
                cam["frames"] = len(cam["frame_files"])
                cam.pop("seconds", None)
            leg.update(link_next=[], extra_frames=[], link_dropped=why)
            solve_leg(leg, refine)
        run_secs[f"first pass {leg['id']}"] = time.time() - t1
    run_secs["first_pass"] = time.time() - t0

    if share_focal and len(legs) > 1 and refine == "all" and focal is None:
        per = {}
        for leg in legs:
            for cam in leg["cameras"]:
                if cam.get("model") and not p.pano(cam["folder"]):
                    fn, _, _ = model_focal(p.reconstruction / cam["model"])
                    per.setdefault(cam["folder"], []).append((fn, cam["registered"], leg["id"]))
                    cam["focal_first_pass"] = round(fn, 4)
        shared = {f: weighted_median([v for v, _, _ in vals], [w for _, w, _ in vals])
                  for f, vals in per.items() if len(vals) > 1}
        if shared:
            print("Shared focal per camera (median of the legs' refined focals, weighted by frames):")
            for f, v in shared.items():
                print(f"  {f}: {v:.3f} (legs: " + ", ".join(f"{l} {x:.3f}" for x, _, l in per[f]) + ")")
            doc["shared_focal"] = {f: round(v, 4) for f, v in shared.items()}
            t0 = time.time()
            for leg in legs:
                solve_leg(leg, "distortion", shared, reuse=True)
            run_secs["shared_focal_pass"] = time.time() - t0
    run_secs["total"] = time.time() - t_start
    doc["seconds"] = {k: round(v, 1) for k, v in run_secs.items()}
    print("reconstruct time: " + ", ".join(f"{k} {v:.0f} s" for k, v in run_secs.items() if not k.startswith("first pass leg")))
    p.legs_json.write_text(json.dumps(doc, indent=1))
    print(f"Wrote {p.legs_json}")
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("--max-turn", type=float, default=35.0, help="Split legs at turns sharper than this (deg)")
    parser.add_argument("--trim-deg", type=float, default=20.0,
                        help="Keep leg frames heading within this of the leg (deg; 180 with --max-turn 180: one leg)")
    parser.add_argument("--max-frames", type=int, default=60, help="Longest leg (frames)")
    parser.add_argument("--min-frames", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0, help="Random seed of COLMAP's mappers (fixed: reruns repeat)")
    parser.add_argument("--leg-overlap", type=int, default=8,
                        help="Run each leg on into the next one's first frames (shared frames that tie the legs together; 0: off)")
    parser.add_argument("--no-shared-focal", action="store_true",
                        help="Keep each leg's own refined focal (by default the legs are solved again with one focal per camera)")
    parser.add_argument("--solo-cameras", action="store_true",
                        help="Solve each camera of a rig on its own (by default a leg's rig cameras, siblings taking frames "
                             "at the same moments as the primary, are solved together with a fixed pose of each camera in the rig)")
    parser.add_argument("--rig", action="store_true", help=argparse.SUPPRESS)  # the default now; kept for old commands
    parser.add_argument("--match-overlap", type=int,
                        help="Match each frame with its next N frames (default: 6 for frames close together, matched by "
                             "nearest neighbours; 10 for frames further apart, matched by LightGlue)")
    parser.add_argument("--ransac-trials", type=int,
                        help="Most RANSAC tries COLMAP's geometric verification spends on a pair (default: 2000 for "
                             "frames close together; 10000, COLMAP's default, for frames further apart)")
    parser.add_argument("--gap-factor", type=float, default=3.0,
                        help="Cut each side or rear camera of a leg to its longest stretch without a jump between frames "
                             f"over this x its median spacing (and over {GAP_FLOOR_M:g} m); 0: off")
    parser.add_argument("--tag", help="Separate run: data/reconstruction_<tag> etc. (GeoCalib's calibration is shared)")
    parser.add_argument("--mapper", choices=["global", "incremental"], default="global",
                        help="COLMAP's global mapper with gravity priors (default) or its incremental mapper")
    parser.add_argument("--split-weak", action="store_true",
                        help="Drop weakly anchored frames, solve again, and let align.py fit each side of the gap "
                             "separately (off by default)")
    parser.add_argument("--matcher", choices=["auto", *MATCHERS], default="auto",
                        help=f"auto (default): nn for frames under {NN_MAX_SPACING:g} m apart, lightglue otherwise")
    parser.add_argument("--calibration", choices=["geocalib", "mapillary"], default="geocalib",
                        help="Start the camera from GeoCalib's calibration (default) or Mapillary's")
    parser.add_argument("--fix-intrinsics", action="store_true",
                        help="Keep the calibration's focal and distortion fixed (bundle adjustment refines them by default)")
    parser.add_argument("--fix-focal", action="store_true",
                        help="Keep the focal fixed but refine the distortion (forward-only legs: the refined focal "
                             "drifts and stretches the model along the street)")
    parser.add_argument("--focal", type=float,
                        help="Start (or with --fix-focal hold) the focal at this fraction of the image's long side "
                             "instead of GeoCalib's, e.g. a camera's value from other reconstructions")
    args = parser.parse_args()
    SEED["value"] = args.seed
    for m in MATCH.values():
        m.update({k: v for k, v in (("overlap", args.match_overlap), ("ransac_trials", args.ransac_trials)) if v})
    reconstruct_roi(args.roi_dir, args.max_turn, args.trim_deg, min_frames=args.min_frames, max_frames=args.max_frames,
                    tag=args.tag, matcher=args.matcher, split_weak=args.split_weak, mapper=args.mapper,
                    calibration=args.calibration, focal=args.focal, gap_factor=args.gap_factor, rig=not args.solo_cameras,
                    leg_overlap=args.leg_overlap, share_focal=not args.no_shared_focal,
                    refine="none" if args.fix_intrinsics else "distortion" if args.fix_focal else "all")


if __name__ == "__main__":
    main()
