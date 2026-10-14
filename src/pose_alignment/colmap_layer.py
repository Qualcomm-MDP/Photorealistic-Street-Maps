"""Stage 4 (COLMAP): sparse reconstruction of every ROI sequence, leg by leg.

  1. Legs: the primary sequence of each line is split at turns sharper than --max-turn
     (with sparse street frames COLMAP cannot follow a corner: it splits or folds there),
     each piece is trimmed to the frames heading within --trim-deg of its median heading
     (drops the curved frames at the corner) and pieces shorter than --min-frames are
     skipped. Legs longer than --max-frames are cut into consecutive chunks at full frame
     density (thinning frames instead makes COLMAP lose track). Every camera of the rig
     gets the frames captured in each leg's time window.
  2. COLMAP per leg and camera (frames linked into <leg>/<sequence folder>/images/):
     SIFT features with one shared camera, Mapillary's calibration held fixed (focal,
     principal point, radial k1/k2 from cameras.json); sequential matching without loop
     detection; the incremental mapper with relaxed thresholds for forward motion (two-view
     tracks kept, 1 deg triangulation angle, looser pose inlier limits). If the mapper still
     splits the frames into pieces, they are merged on shared frames, bundle adjusted and
     re-triangulated (flagged, a merged model can keep a seam); --largest-piece uses the
     largest single piece instead.
     Feature masks: burned-in overlays (timestamp/GPS text, bonnet) are found as image rows
     that stay sharp and identical across frames, and ignored. --mask-moving also ignores
     the segmentation's sky/vehicle/person pixels (moving cars give matches that do not
     belong to the static scene).
  3. Point labels: each 3D point is labelled building or road by majority vote over the
     segmentation mask pixels it was observed at.
  4. Calibration check: Mapillary's focal is sometimes badly wrong (an iPhone at 2800 px
     instead of ~1600), which squashes the model. The camera height above a plane fitted to
     the road points (in metres via Mapillary's track) flags it: if it is outside
     --height-range, or few frames registered, and the camera faces roughly forward (sees
     both sides of the street, so OSM can judge it), COLMAP is rerun self-calibrating from
     a generic focal, both versions are aligned to OSM on their own, and the one whose
     facades fit the walls better is kept.
  5. A 3D view of each sparse model: points in their colours, a plan view coloured by label
     (building / road / other) and the camera path, in approximate metres.

Outputs in rois/mapillary_roi_N/data/colmap/:
  legs.json                               legs, cameras, models, camera heights, calibration
  <leg>/<sequence folder>/sparse/<k>/     COLMAP models (binary, txt/ export, points.ply)
  <leg>/<sequence folder>/colmap.log      COLMAP output
3D views go to rois/mapillary_roi_N/results/sparse_views/<leg>_<sequence folder>.png.

Usage:
  python colmap_layer.py rois/mapillary_roi_1
  python colmap_layer.py rois/mapillary_roi_1 --mask-moving --force
"""

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (MASK_BUILDING, MASK_ROAD, MASK_SKY_OR_MOVING, LocalFrame, RoiPaths, gravity_frame, label_points,
                    load_cameras, load_points, model_size, overpass, similarity_2d, straight_stretches)

LEG_SETTINGS = ("max_turn", "trim_deg", "min_frames", "max_frames")


# ----------------------------------------------------------------------------- legs

def street_name(lon, lat):
    """Name of the named street nearest to a point (OSM), for labelling legs."""
    try:
        data = overpass(f'[out:json][timeout:30];way(around:25,{lat},{lon})["highway"]["name"];out geom;')
    except SystemExit:
        return "unknown street"
    frame = LocalFrame(lon, lat)
    best = (float("inf"), "unnamed street")
    for way in data["elements"]:
        pts = [frame.xy(g["lon"], g["lat"]) for g in way.get("geometry", [])]
        for a, b in zip(pts, pts[1:]):
            d = np.subtract(b, a)
            t = np.clip(-np.dot(a, d) / max(np.dot(d, d), 1e-12), 0, 1)
            best = min(best, (float(np.hypot(*(np.add(a, t * d)))), way["tags"]["name"]))
    return best[1]


def plan_legs(paths, sequences, frames, settings):
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
            mid = core[len(core) // 2]
            pts = [frame.xy(f["lon"], f["lat"]) for f in core]
            cams = []
            for s in seqs:
                window = sorted(f["file"] for f in frames[s["folder"]].values() if t0 <= f["captured_at_ms"] <= t1)
                if len(window) >= 10:
                    cams.append({"folder": s["folder"], "sequence": s["sequence"], "role": s["role"],
                                 "view_deg": s["view_deg"], "frames": len(window), "frame_files": window})
            leg = {"id": leg_id, "line": line, "street": street_name(mid["lon"], mid["lat"]),
                   "primary_frames": len(core), "length_m": round(sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))),
                   "cameras": cams}
            print(f"  {leg_id}: line {line}, {leg['street']}, {leg['length_m']} m, "
                  + ", ".join(f"{c['folder']} {c['frames']} frames" for c in cams))
            legs.append(leg)
    if not legs:
        raise SystemExit("No straight stretch with enough frames; lower --min-frames or draw a longer route")
    return legs


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


def num_images(model, log):
    return int(next(l for l in colmap(log, "model_analyzer", "--path", model).stderr.splitlines()
                    if "Registered images:" in l).split(":")[-1])


def merge_models(models, work, database, images, fixed_intrinsics, log):
    """Merge overlapping models into sparse/merged, then bundle adjust and re-triangulate.

    model_merger aligns two models on their shared frames. Bundle adjustment then refines
    all poses together, and point_triangulator rebuilds every 3D point from the matches
    with the final poses (bundle_adjuster alone leaves stale per-point errors behind)."""
    models = sorted(models, key=lambda m: num_images(m, log), reverse=True)
    current = models[0]
    for i, other in enumerate(models[1:]):
        out = work / f"merge_{i}"
        out.mkdir(parents=True)
        result = colmap(log, "model_merger", "--input_path1", current, "--input_path2", other,
                        "--output_path", out, check=False)
        if result.returncode == 0 and "Merge succeeded" in result.stdout + result.stderr:
            current = out
    if current == models[0]:
        return None
    fixed = 0 if fixed_intrinsics else 1
    adjusted = work / "adjusted"
    adjusted.mkdir()
    colmap(log, "bundle_adjuster", "--input_path", current, "--output_path", adjusted,
           "--BundleAdjustment.refine_focal_length", fixed, "--BundleAdjustment.refine_extra_params", fixed)
    merged = work.parent / "merged"
    merged.mkdir()
    colmap(log, "point_triangulator", "--database_path", database, "--image_path", images,
           "--input_path", adjusted, "--output_path", merged, "--clear_points", 1, "--refine_intrinsics", 0)
    return merged


def run_colmap(images, out, camera_params, fix_intrinsics, mask_dir=None, overlap=10, use_gpu=0):
    """Features, sequential matching and the relaxed incremental mapper; every model is
    exported to txt/ and points.ply. Returns the sparse folder."""
    database, sparse, log = out / "database.db", out / "sparse", out / "colmap.log"
    database.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    colmap(log, "feature_extractor", "--database_path", database, "--image_path", images,
           "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "OPENCV",
           "--ImageReader.camera_params", camera_params, "--FeatureExtraction.use_gpu", use_gpu,
           *(["--ImageReader.mask_path", mask_dir] if mask_dir else []))
    colmap(log, "sequential_matcher", "--database_path", database, "--SequentialMatching.overlap", overlap,
           "--SequentialMatching.loop_detection", 0, "--FeatureMatching.use_gpu", use_gpu)
    # Forward motion: points near the direction of travel triangulate at small angles and
    # most tracks are short, so COLMAP's defaults (ignore two-view tracks, 1.5 deg minimum
    # angle) leave too few 3D points to register the next frame and the model splits.
    colmap(log, "mapper", "--database_path", database, "--image_path", images, "--output_path", sparse,
           *(["--Mapper.ba_refine_focal_length", 0, "--Mapper.ba_refine_extra_params", 0] if fix_intrinsics else []),
           "--Mapper.tri_ignore_two_view_tracks", 0, "--Mapper.tri_min_angle", 1.0,
           "--Mapper.filter_min_tri_angle", 1.0, "--Mapper.abs_pose_min_num_inliers", 15,
           "--Mapper.abs_pose_min_inlier_ratio", 0.1, "--Mapper.init_min_tri_angle", 4,
           "--Mapper.init_max_forward_motion", 0.99)
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if len(models) > 1:
        merged = merge_models(models, sparse / "_merge_work", database, images, fix_intrinsics, log)
        shutil.rmtree(sparse / "_merge_work", ignore_errors=True)
        if merged:
            models.append(merged)
    for model in models:
        (model / "txt").mkdir(exist_ok=True)
        colmap(log, "model_converter", "--input_path", model, "--output_path", model / "txt", "--output_type", "TXT")
        colmap(log, "model_converter", "--input_path", model, "--output_path", model / "points.ply", "--output_type", "PLY")
    return sparse


def best_model(sparse, allow_merged=True):
    """(model with the most registered frames, sizes of COLMAP's pieces), or (None, [])."""
    models = [d for d in sparse.iterdir() if d.is_dir()] if sparse.exists() else []
    pieces = sorted((model_size(d) for d in models if d.name != "merged"), reverse=True)
    if not allow_merged:
        models = [d for d in models if d.name != "merged"] or models
    if not models:
        return None, []
    return max(models, key=model_size), pieces


def detect_overlay_rows(folder, sample=15, band=0.15):
    """Rows at the bottom of the frames covered by a static overlay, or 0.

    Overlay text and bonnets stay sharp and identical in every frame, while the scene
    behind them changes. A row is overlay if enough of its pixels are both static across
    frames and on an edge; the mask covers the bottom band from the highest such row."""
    paths = sorted(folder.glob("frame_*.jpg"))
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(sample, len(paths))).astype(int)]
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
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


def write_feature_masks(names, images, mask_dir, bottom_rows, seg_dir=None):
    """COLMAP feature masks (<image name>.png, black = ignore): the bottom overlay rows and,
    with seg_dir, the segmentation's sky/vehicle/person pixels. Returns mask_dir or None."""
    if not bottom_rows and seg_dir is None:
        return None
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir(parents=True)
    for name in names:
        h, w = cv2.imread(str(images / name)).shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        if bottom_rows:
            mask[h - bottom_rows:] = 0
        if seg_dir is not None:
            seg = cv2.imread(str(seg_dir / f"{Path(name).stem}.png"), cv2.IMREAD_GRAYSCALE)
            if seg is not None:
                mask[seg == MASK_SKY_OR_MOVING] = 0
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask)
    return mask_dir


def mapillary_camera_params(first, width, height):
    """COLMAP OPENCV params from Mapillary's calibration ([focal, k1, k2], focal normalised by
    the larger image side; radial distortion on normalised coordinates), or None."""
    params = first.get("camera_parameters")
    if not params or len(params) < 3:
        return None
    focal, k1, k2 = params[:3]
    f = focal * max(width, height)
    return ",".join(f"{v:.6g}" for v in (f, f, width / 2, height / 2, k1, k2, 0, 0))


# ----------------------------------------------------------------------------- checks and views

def model_scale(model_txt, frames):
    """(names, centres, rots, origin, basis, metres per model unit) from Mapillary's track."""
    names, centres, rots = load_cameras(model_txt)
    origin, basis = gravity_frame(centres, rots)
    lev = (centres - origin) @ basis.T
    f0 = frames[names[0]]
    enu = LocalFrame(f0["lon"], f0["lat"])
    gps = np.array([enu.xy(frames[n]["lon"], frames[n]["lat"]) for n in names])
    return names, centres, rots, origin, basis, similarity_2d(lev[:, :2], gps)[0]


def camera_height(model_txt, frames, mask_dir, near_m=25.0, min_points=30):
    """Median camera height (m) above a plane fitted (RANSAC) to the road points near the
    path, using Mapillary's scale; None if there are too few road points."""
    names, centres, rots, origin, basis, scale = model_scale(model_txt, frames)
    xyz, _, _ = load_points(model_txt)
    road, _ = label_points(model_txt, mask_dir, MASK_ROAD)
    P = xyz[road]
    if len(P) < min_points:
        return None
    P = P[np.min(np.linalg.norm(P[:, None] - centres[None, ::2], axis=2), axis=1) * scale < near_m]
    if len(P) < min_points:
        return None
    rng = np.random.default_rng(0)
    thresh = 0.15 / scale
    best = None
    for _ in range(1000):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_points:
        return None
    Q = P[best]
    centre = Q.mean(axis=0)
    n = np.linalg.svd(Q - centre)[2][2]
    return float(abs(np.median((centres - centre) @ n)) * scale)


def plot_sparse(path, model_txt, frames, mask_dir, title):
    """3D view (point colours) and plan view (labels) of a model with its camera path,
    levelled by the travel direction and scaled to metres with Mapillary's track."""
    names, centres, rots, origin, basis, scale = model_scale(model_txt, frames)
    xyz, rgb, track = load_points(model_txt)
    cam = (centres - origin) @ basis.T * scale
    pts = (xyz - origin) @ basis.T * scale
    near = np.min(np.linalg.norm(pts[:, None, :2] - cam[None, ::3, :2], axis=2), axis=1) < 50
    building = label_points(model_txt, mask_dir, MASK_BUILDING)[0]
    road = label_points(model_txt, mask_dir, MASK_ROAD)[0]
    # Ground to rooftops only: a few far, badly triangulated points would stretch the 3D box.
    keep = near & (track >= 2) & (pts[:, 2] > -5) & (pts[:, 2] < 35)
    pts, rgb, building, road = pts[keep], rgb[keep], building[keep], road[keep]
    look = rots[:, :, 2] @ basis.T

    fig = plt.figure(figsize=(18, 8.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=1.2, linewidths=0, depthshade=False)
    ax.plot(cam[:, 0], cam[:, 1], cam[:, 2], "-", color="crimson", lw=2.5, label="camera path")
    ax.scatter(cam[::5, 0], cam[::5, 1], cam[::5, 2], color="crimson", s=10)
    ax.set_xlabel("right [m]")
    ax.set_ylabel("forward [m]")
    ax.set_zlabel("up [m]")
    ax.view_init(elev=22, azim=-135)
    if len(pts):
        ax.set_box_aspect(np.maximum(np.ptp(np.vstack([pts, cam]), axis=0), 1.0))
    ax.set_title("3D view (point colours from the photos)")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(1, 2, 2)
    other = ~(building | road)
    ax.scatter(pts[other, 0], pts[other, 1], s=1, c="0.7", linewidths=0, label="other")
    ax.scatter(pts[road, 0], pts[road, 1], s=1.5, c="mediumpurple", linewidths=0, label="road")
    ax.scatter(pts[building, 0], pts[building, 1], s=1.5, c="darkorange", linewidths=0, label="building")
    ax.plot(cam[:, 0], cam[:, 1], "-", color="crimson", lw=1.5, label="camera path")
    step = 0.04 * max(np.ptp(cam[:, 1]), 20.0)  # arrow length
    ax.quiver(cam[::5, 0], cam[::5, 1], look[::5, 0], look[::5, 1], color="crimson",
              angles="xy", scale_units="xy", scale=1 / step, width=0.003)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("right [m]")
    ax.set_ylabel("forward [m]")
    ax.set_title("Plan view by segmentation label (arrows: viewing direction)")
    ax.legend(loc="upper right", markerscale=6)
    fig.suptitle(f"{title}: {len(xyz)} points, {len(cam)} cameras (approximate metres from Mapillary's track)")
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- whole ROI

def reconstruct_roi(roi_dir, max_turn=35.0, trim_deg=20.0, min_frames=15, max_frames=60, mask_moving=False,
                    largest_piece=False, height_range=(0.9, 3.0), force=False):
    """Plan the legs and run COLMAP on every leg and camera; writes legs.json."""
    paths = RoiPaths(roi_dir)
    sequences = paths.sequences()
    frames = {s["folder"]: paths.frames(s["folder"]) for s in sequences}
    for s in sequences:
        if not (paths.masks / s["folder"]).exists():
            raise SystemExit(f"No masks for {s['folder']} in {paths.masks}: run the segmentation stage first")
    settings = {"max_turn": max_turn, "trim_deg": trim_deg, "min_frames": min_frames, "max_frames": max_frames}
    paths.colmap.mkdir(parents=True, exist_ok=True)
    old = json.loads(paths.legs_json.read_text()) if paths.legs_json.exists() else None
    if old and not force and old.get("settings") == settings:
        legs = old["legs"]
        print(f"Using the {len(legs)} legs planned in {paths.legs_json}")
    else:
        print("Planning legs:")
        legs = plan_legs(paths, sequences, frames, settings)
    doc = {"settings": settings, "mask_moving": mask_moving, "legs": legs}
    lo, hi = height_range
    overlays = {}

    for leg in legs:
        for cam in leg["cameras"]:
            folder, meta = cam["folder"], frames[cam["folder"]]
            work = paths.colmap / leg["id"] / folder
            images = work / "images"
            seg = paths.masks / folder
            label = f"{leg['id']}/{folder}"
            sparse = work / "sparse"
            w, h = cv2.imread(str(paths.images / folder / cam["frame_files"][0])).shape[1::-1]
            params = mapillary_camera_params(meta[cam["frame_files"][0]], w, h)
            # Without Mapillary's calibration, self-calibrate from a generic focal.
            cam["calibration"] = "mapillary" if params else "self-calibrated"
            params = params or f"{0.8 * max(w, h)},{0.8 * max(w, h)},{w / 2},{h / 2},0,0,0,0"
            if force or not any(model_size(d) for d in (sparse.iterdir() if sparse.exists() else [])):
                if images.exists():
                    shutil.rmtree(images)
                images.mkdir(parents=True)
                # Hard links (no extra disk space): COLMAP resolves symlinks and would record
                # the frames under their link targets' paths instead of their names.
                for name in cam["frame_files"]:
                    try:
                        os.link(paths.images / folder / name, images / name)
                    except OSError:
                        shutil.copy2(paths.images / folder / name, images / name)
                if folder not in overlays:
                    overlays[folder] = detect_overlay_rows(paths.images / folder)
                cam["mask_bottom"] = overlays[folder]
                masks = write_feature_masks(cam["frame_files"], images, work / "feature_masks", cam["mask_bottom"],
                                            seg if mask_moving else None)
                print(f"{label}: COLMAP on {cam['frames']} frames ({cam['calibration']} calibration"
                      f"{', overlay mask %d px' % cam['mask_bottom'] if cam['mask_bottom'] else ''}"
                      f"{', sky/moving masked' if mask_moving else ''})")
                try:
                    run_colmap(images, work, params, cam["calibration"] == "mapillary", masks)
                except RuntimeError as e:
                    print(f"  {e}")
            model, pieces = best_model(sparse, not largest_piece)
            cam["colmap_pieces"] = pieces
            if model is None or model_size(model) < 3:
                cam.update(model=None, registered=0, camera_height_m=None)
                print(f"{label}: COLMAP registered no model")
                continue
            height = camera_height(model / "txt", meta, seg)
            print(f"{label}: {model_size(model)}/{cam['frames']} registered{f' (pieces {pieces})' if len(pieces) > 1 else ''}, "
                  f"camera height {'?' if height is None else f'{height:.2f} m'}")

            # Calibration check, forward cameras only: a side camera sees one wall line, which a
            # wrong scale plus a shift fits just as well, so OSM cannot judge its calibration.
            forward = cam["view_deg"] is not None and abs(cam["view_deg"]) <= 35
            suspect = (height is not None and not lo <= height <= hi) or model_size(model) < 0.8 * cam["frames"]
            if suspect and forward and cam["calibration"] == "mapillary":
                model, height = calibration_trial(cam, model, height, work, images, meta, seg, largest_piece,
                                                  height_range, force)
            elif suspect and not forward:
                print(f"{label}: height or registration off, but a side-facing camera cannot be checked against "
                      "OSM; keeping Mapillary's calibration")
            # Relative to the colmap folder, so the ROI folder can be moved.
            cam.update(model=str((model / "txt").relative_to(paths.colmap)), registered=model_size(model), camera_height_m=height,
                       merged_model=model.name == "merged")
            view = paths.results / "sparse_views" / f"{leg['id']}_{folder}.png"
            view.parent.mkdir(parents=True, exist_ok=True)
            if force or not view.exists() or view.stat().st_mtime < (model / "txt" / "points3D.txt").stat().st_mtime:
                plot_sparse(view, model / "txt", meta, seg, label)
        paths.legs_json.write_text(json.dumps(doc, indent=1))
    paths.legs_json.write_text(json.dumps(doc, indent=1))
    print(f"\nWrote {paths.legs_json}; 3D views in {paths.results / 'sparse_views'}")
    return doc


def calibration_trial(cam, model, height, work, images, meta, seg, largest_piece, height_range, force):
    """Rerun COLMAP self-calibrating, align both versions to OSM on their own and keep the one
    whose facades fit the walls better. Returns (model, height)."""
    from osm_alignment_layer import AlignmentError, OSMAlignmentLayer

    lo, hi = height_range
    work2 = work.parent / f"{work.name}_selfcal"
    w, h = cv2.imread(str(images / cam["frame_files"][0])).shape[1::-1]
    f0 = 0.8 * max(w, h)
    if force or not best_model(work2 / "sparse")[0]:
        print(f"  calibration check: rerunning COLMAP self-calibrating from a {f0:.0f} px focal")
        work2.mkdir(parents=True, exist_ok=True)
        try:
            run_colmap(images, work2, f"{f0},{f0},{w / 2},{h / 2},0,0,0,0", False,
                       (work / "feature_masks") if (work / "feature_masks").exists() else None)
        except RuntimeError as e:
            print(f"  {e}")
    model2, _ = best_model(work2 / "sparse", not largest_piece)
    if model2 is None or model_size(model2) < 0.8 * cam["frames"]:
        print("  self-calibrated model registered too few frames; keeping Mapillary's calibration")
        return model, height
    height2 = camera_height(model2 / "txt", meta, seg)
    fits = {}
    for label, mdl, hgt in (("mapillary", model, height), ("self-calibrated", model2, height2)):
        h_use = hgt if hgt is not None and lo <= hgt <= hi else 1.4
        try:
            res = OSMAlignmentLayer(h_use).align([{"label": cam["folder"], "model": mdl / "txt", "frames": meta,
                                                   "masks": seg}])
            fits[label] = res["stats"]["facade_to_wall_aligned"]["all"]["median_m"]
        except AlignmentError as e:
            fits[label] = float("inf")
            print(f"  {label} trial not aligned: {e}")
    print("  OSM facade fit, median facade-to-wall: " + ", ".join(f"{k} {v:.2f} m" for k, v in fits.items()))
    cam["calibration_trial"] = fits
    if fits["self-calibrated"] < 0.8 * fits["mapillary"]:
        cam["calibration"] = "self-calibrated"
        return model2, height2
    return model, height


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roi_dir", type=Path, help="ROI folder (rois/mapillary_roi_N) after the segmentation stage")
    parser.add_argument("--max-turn", type=float, default=35.0, help="Split legs at turns sharper than this (deg)")
    parser.add_argument("--trim-deg", type=float, default=20.0, help="Keep leg frames heading within this of the leg")
    parser.add_argument("--min-frames", type=int, default=15, help="Skip legs with fewer primary frames")
    parser.add_argument("--max-frames", type=int, default=60, help="Cut longer legs into chunks of at most this many frames")
    parser.add_argument("--mask-moving", action="store_true",
                        help="Also ignore features on sky, vehicles and people (segmentation masks)")
    parser.add_argument("--largest-piece", action="store_true",
                        help="When COLMAP splits a leg, use its largest piece instead of the merged model")
    parser.add_argument("--height-range", type=float, nargs=2, default=[0.9, 3.0],
                        help="Plausible camera heights (m); outside it the calibration is checked")
    parser.add_argument("--force", action="store_true", help="Re-plan the legs and rerun COLMAP everywhere")
    args = parser.parse_args()
    reconstruct_roi(args.roi_dir, args.max_turn, args.trim_deg, args.min_frames, args.max_frames, args.mask_moving,
                    args.largest_piece, args.height_range, args.force)


if __name__ == "__main__":
    main()
