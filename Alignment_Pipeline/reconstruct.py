"""Stage 5 (reconstruct): camera poses and sparse points with ALIKED features and COLMAP 4.2.1's
global mapper (GLOMAP), using GeoCalib's calibration and gravity.

Legs: the primary sequence of each drawn line is split at turns sharper than --max-turn (its
frames heading within --trim-deg of the leg kept) and into chunks of at most --max-frames; each
rig camera (sibling sequence) gets its frames in the leg's time window. --max-turn 180
--trim-deg 180 --max-frames 200 keeps a whole route in one leg.

Per leg and sequence:
  1. Frames hard-linked into the work folder; feature masks for the burned-in overlay and for sky,
     vehicles and people from the segmentation.
  2. Camera (camera_setup): one per sequence from GeoCalib's focal, principal point at the
     centre; RADIAL_FISHEYE for strongly distorted lenses, RADIAL otherwise. A warning when
     Mapillary's focal differs by more than 10%.
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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import MASK_SKY_OR_MOVING, LocalFrame, RoiPaths, load_cameras, model_size, straight_stretches

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
# Nearest-neighbour matching is ~15x faster than LightGlue and as good on frames up to ~5.6 m
# apart (roi_2, 7, 8, 11, main_street_east); on frames 8-10 m apart (roi_4, 5, 9) it finds too
# few matches between neighbours and the reconstruction becomes unstable (bends, lost frames)
NN_MAX_SPACING = 6.0  # metres between frames
WEAK_RATIO = 0.35          # frames under this x the leg's median 3D observations are dropped
DISAGREE = 0.10            # warn when Mapillary's and GeoCalib's focals are further apart than this


# ----------------------------------------------------------------------------- legs

def plan_legs(p, sequences, frames, settings):
    """Straight legs of each line's primary sequence, with every rig camera's frames in each
    leg's time window."""
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
        for core in chunks:
            leg_id = f"leg{len(legs) + 1:02d}"
            t0, t1 = core[0]["captured_at_ms"] - 500, core[-1]["captured_at_ms"] + 500
            pts = [frame.xy(f["lon"], f["lat"]) for f in core]
            cams = []
            for s in seqs:
                window = sorted(f["file"] for f in frames[s["folder"]].values() if t0 <= f["captured_at_ms"] <= t1)
                if len(window) >= 10:
                    cams.append({"folder": s["folder"], "sequence": s["sequence"], "role": s["role"],
                                 "view_deg": s["view_deg"], "frames": len(window), "frame_files": window})
            leg = {"id": leg_id, "line": line, "primary_frames": len(core),
                   "length_m": round(sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))), "cameras": cams}
            print(f"  {leg_id}: line {line}, {leg['length_m']} m, "
                  + ", ".join(f"{c['folder']} {c['frames']} frames" for c in cams))
            legs.append(leg)
    if not legs:
        raise SystemExit("No straight stretch with enough frames; lower --min-frames or draw a longer route")
    return legs


# ----------------------------------------------------------------------------- camera

def mapillary_calibration(p, folder):
    """(make, the sequence's most common Mapillary [focal, k1, k2]) or (make, None)."""
    frames = list(p.frames(folder).values())
    make = next((f.get("make") for f in frames if f.get("make")), "") or ""
    cands = Counter(tuple(f["camera_parameters"]) for f in frames if len(f.get("camera_parameters") or []) == 3)
    return make, (list(cands.most_common(1)[0][0]) if cands else None)


def camera_setup(p, folder, k1_limit=-0.2):
    """(COLMAP camera model, params, description) for a sequence, from GeoCalib's median focal:
    RADIAL_FISHEYE (f, k1, k2 on the angle from the axis, starting undistorted) for a strongly
    distorted lens (GeoCalib k1 below k1_limit, e.g. wide action cameras), RADIAL (f, k1, k2,
    starting from GeoCalib's k1) otherwise. Bundle adjustment refines it. On roi_11's wide
    motorcycle camera the refined fisheye model kept the scale of a leg's second half at 0.93 of
    its first, RADIAL at 0.80-0.84.
    Mapillary's focal is recorded for comparison only: starting smartphones from it (roi_2, 4, 5,
    7, 8, 9) changed nothing where frames are close and did not steady widely spaced ones (roi_9
    collapsed), and it is sometimes far off (roi_15: 1.338)."""
    gc = json.loads(p.calibration_json(folder).read_text())
    w, h = gc["width"], gc["height"]
    fisheye = gc["k1"] < k1_limit
    model = "RADIAL_FISHEYE" if fisheye else "RADIAL"
    params = [gc["focal_px"], w / 2, h / 2, 0.0 if fisheye else gc["k1"], 0.0]
    make, mly = mapillary_calibration(p, folder)
    info = {"source": "geocalib", "model": model, "params": params, "make": make,
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


def reconstruct_camera(p, leg, cam, overlays, log_prefix, matcher="auto", split_weak=False, mapper="global"):
    folder = cam["folder"]
    work = p.reconstruction / leg["id"] / folder
    images, sparse, log, db = work / "images", work / "sparse", work / "colmap.log", work / "database.db"
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
    cmodel, cparams, cinfo = camera_setup(p, folder)
    if cinfo.get("focal_disagreement") is not None and abs(cinfo["focal_disagreement"]) > DISAGREE:
        print(f"{log_prefix}: WARNING: GeoCalib's focal {cinfo['geocalib_focal_norm']:.3f} and Mapillary's "
              f"{cinfo['mapillary_focal_norm']:.3f} (of the long side) differ by "
              f"{100 * cinfo['focal_disagreement']:+.0f}%; starting from GeoCalib's", flush=True)
    spacing = frame_spacing(p, folder, cam["frame_files"])
    if matcher == "auto":
        matcher = "nn" if spacing < NN_MAX_SPACING else "lightglue"
    cam.update(matcher=matcher, frame_spacing_m=round(spacing, 2))
    print(f"{log_prefix}: {cam['frames']} frames {spacing:.1f} m apart, ALIKED + {MATCHERS[matcher]}, "
          f"{cmodel} from GeoCalib, {mapper} mapper", flush=True)
    params = ",".join(f"{v:.8g}" for v in cparams)
    script = "learned_features.py" if matcher == "lightglue" else "nn_features.py"
    run_python(log, script, "--images", images, "--out", work / "learned", "--overlap", 10, "--masks", masks)
    colmap(log, "feature_importer", "--database_path", db, "--image_path", images, "--import_path",
           work / "learned" / "features", "--ImageReader.single_camera", 1, "--ImageReader.camera_model",
           cmodel, "--ImageReader.camera_params", params)
    colmap(log, "matches_importer", "--database_path", db, "--match_list_path", work / "learned" / "matches.txt",
           "--match_type", "raw", "--FeatureMatching.use_gpu", 0)
    gravity = {n: (-np.array(v["up_cam"])).tolist() for n, v in gc["frames"].items() if n in cam["frame_files"]}
    n_priors = write_gravity_priors(db, gravity)

    def solve(image_list=None, use=mapper):
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir()
        if use == "global":
            colmap(log, "global_mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                   "--GlobalMapper.ra_use_gravity", 1, "--GlobalMapper.ba_refine_focal_length", 1,
                   "--GlobalMapper.ba_refine_extra_params", 1, "--GlobalMapper.ba_refine_principal_point", 0,
                   *(["--GlobalMapper.image_list_path", image_list] if image_list else []))
        else:
            colmap(log, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                   "--Mapper.ba_refine_focal_length", 1, "--Mapper.ba_refine_extra_params", 1, *INCREMENTAL,
                   *(["--Mapper.image_list_path", image_list] if image_list else []))
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
        return best_model(sparse)

    model, pieces = solve()
    image_list, dropped = None, {}
    if split_weak and model is not None and model_size(model) >= 3:
        # Frames the solve barely anchors (e.g. a parked truck filling the view: its matches are
        # rejected and it hides the buildings) get placed badly and can split the leg into parts at
        # different scales. Drop those with under WEAK_RATIO x the leg's median observation count
        # and solve the leg again without them (the matches across the gap keep it in one piece).
        counts = observations_per_image(model / "txt")
        med = float(np.median(list(counts.values())))
        dropped = {n: c for n, c in counts.items() if c < WEAK_RATIO * med}
        if dropped:
            keep = [n for n in cam["frame_files"] if n not in dropped]
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
    cam.update(mapper_used=used, mapper_fallback=fallback)
    cam["dropped_frames"] = {n: int(c) for n, c in sorted(dropped.items())}
    cam.update(calibration={**cinfo, "refined": True}, gravity_priors=n_priors, colmap_pieces=pieces,
               mask_bottom=overlays[folder])
    if model is None or model_size(model) < 3:
        cam.update(model=None, registered=0)
        print(f"{log_prefix}: no model")
        return
    cam.update(model=str((model / "txt").relative_to(p.reconstruction)), registered=model_size(model))
    print(f"{log_prefix}: {cam['registered']}/{cam['frames']} registered with the {used} mapper"
          f"{f', {len(dropped)} weak frames dropped' if dropped else ''}"
          f"{f' (pieces {pieces})' if len(pieces) > 1 else ''}, {n_priors} gravity priors")


def reconstruct_roi(roi_dir, max_turn=35.0, trim_deg=20.0, min_frames=15, max_frames=60, tag=None, matcher="auto",
                    split_weak=False, mapper="global"):
    p = RoiPaths(roi_dir, tag)
    sequences = p.sequences()
    frames = {s["folder"]: p.frames(s["folder"]) for s in sequences}
    for s in sequences:
        if not p.calibration_json(s["folder"]).exists():
            raise SystemExit(f"No GeoCalib calibration for {s['folder']}: run the calibrate stage first")
    settings = {"max_turn": max_turn, "trim_deg": trim_deg, "min_frames": min_frames, "max_frames": max_frames}
    p.reconstruction.mkdir(parents=True, exist_ok=True)
    print("Planning legs:")
    legs = plan_legs(p, sequences, frames, settings)
    doc = {"pipeline": "Alignment_Pipeline", "settings": settings, "features": "aliked", "mapper": mapper,
           "refine_intrinsics": True, "mask_moving": True, "gravity_priors": True, "split_weak": split_weak, "weak_frame_ratio": WEAK_RATIO,
           "matcher": matcher, "tag": tag, "legs": legs}
    overlays = {}
    for leg in legs:
        for cam in leg["cameras"]:
            label = f"{leg['id']}/{cam['folder']}"
            try:
                reconstruct_camera(p, leg, cam, overlays, label, matcher, split_weak, mapper)
            except RuntimeError as e:
                cam.update(model=None, registered=0)
                print(f"{label}: failed: {e}")
            p.legs_json.write_text(json.dumps(doc, indent=1))
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
    parser.add_argument("--tag", help="Separate run: data/reconstruction_<tag> etc. (GeoCalib's calibration is shared)")
    parser.add_argument("--mapper", choices=["global", "incremental"], default="global",
                        help="COLMAP's global mapper with gravity priors (default) or its incremental mapper")
    parser.add_argument("--split-weak", action="store_true",
                        help="Drop weakly anchored frames, solve again, and let align.py fit each side of the gap "
                             "separately (off by default)")
    parser.add_argument("--matcher", choices=["auto", *MATCHERS], default="auto",
                        help=f"auto (default): nn for frames under {NN_MAX_SPACING:g} m apart, lightglue otherwise")
    args = parser.parse_args()
    reconstruct_roi(args.roi_dir, args.max_turn, args.trim_deg, min_frames=args.min_frames, max_frames=args.max_frames,
                    tag=args.tag, matcher=args.matcher, split_weak=args.split_weak, mapper=args.mapper)


if __name__ == "__main__":
    main()
