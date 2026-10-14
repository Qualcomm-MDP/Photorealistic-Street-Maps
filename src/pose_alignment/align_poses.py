"""Pipeline stage: camera poses for texturing, picked with main.py --poses.

Runs between `fetch` and `build_mesh` and edits the Mapillary metadata in place, in the
format apply_textures reads (latitude/longitude, world-to-camera `rotvec` in ENU,
focal_length_pixels at original_width x original_height, image_url):

  raw       Mapillary's raw GPS position with its computed rotation (the default, unchanged)
  computed  Mapillary's computed (OpenSfM) position with its computed rotation
  colmap    the alignment layers on the OSM streets in the bounding box (streets_from_bbox ->
            mapillary -> segmentation -> colmap -> osm). Every image the OSM stage marks
            reliable gets its corrected pose and measured camera height; reliable images the
            main fetch did not return are added from the alignment layers' downloads. Other
            images keep their raw Mapillary pose, or are dropped with --aligned-only. A leg
            whose aligned cameras end up more than one corridor width (15 m) from the street
            they were captured on is rejected too: the OSM stage can call a leg reliable when
            a single short facade pinned its heading.
            --alignment-dir reuses an ROI folder from an earlier run, finishing any stage
            that did not complete.

The colmap source writes <ROI folder>/pose_overrides.json: per image the pose used, how far
the camera moved and turned, and why an image kept its Mapillary pose. If the alignment
fails (COLMAP missing, no usable capture, no fit to OSM), the reason is printed and every
image keeps its Mapillary pose: the pipeline carries on.
"""

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geo_common import LocalFrame, RoiPaths, wrap_deg
from .streets_from_bbox import distance_to_polyline

POSE_SOURCES = ("raw", "computed", "colmap")


def _use_computed(mapillary):
    n = 0
    for meta in mapillary.values():
        if meta.get("computed_latitude") is not None:
            meta["latitude"], meta["longitude"] = meta["computed_latitude"], meta["computed_longitude"]
            n += 1
    print(f"  align_poses: Mapillary computed positions for {n} of {len(mapillary)} images")


def _run_alignment(bbox, roi_dir=None):
    """Run the alignment layers (or finish an earlier run in roi_dir); returns its RoiPaths."""
    from .colmap_layer import reconstruct_roi
    from .mapillary_layer import fetch_roi
    from .osm_alignment_layer import align_roi
    from .run_pipeline import stage_done
    from .segmentation_layer import segment_roi
    from .streets_from_bbox import streets_roi

    if roi_dir is None:
        roi = streets_roi((bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat))
        roi_dir = fetch_roi(roi, os.environ["MAPILLARY_ACCESS_TOKEN"])
    paths = RoiPaths(roi_dir)
    done = stage_done(paths)
    if not done["mapillary"]:
        raise RuntimeError(f"{paths.root} has no finished Mapillary download")
    if not done["segmentation"]:
        segment_roi(paths.root)
    if not done["colmap"]:
        colmap_bin = os.environ.get("COLMAP_BIN")
        if colmap_bin:
            os.environ["PATH"] = f"{Path(colmap_bin).parent}{os.pathsep}{os.environ['PATH']}"
        if shutil.which("colmap") is None:
            raise RuntimeError("COLMAP not found: install it, or set COLMAP_BIN in .env to its executable")
        reconstruct_roi(paths.root)
    if not done["osm"]:
        align_roi(paths.root)
    return paths


def _heading(R_world_to_cam):
    """Compass heading (deg) of a camera's viewing direction."""
    forward = np.asarray(R_world_to_cam)[2]
    return math.degrees(math.atan2(forward[0], forward[1]))


def _aligned_cameras(paths):
    """{Mapillary image id: (sequence folder, camera record)} from the OSM stage's output."""
    cameras = {}
    for f in sorted(paths.alignment.glob("*.json")):
        doc = json.loads(f.read_text())
        for cam in doc.get("cameras", []):
            cameras[str(cam["id"])] = (doc["folder"], cam)
    return cameras


def _off_street_legs(paths, cameras, max_off_street_m):
    """{leg: metres} for legs whose aligned cameras stray further than max_off_street_m from
    the ROI's street lines. Cameras drive along the street, so a leg that leaves it was fitted
    to the wrong walls (e.g. a heading pinned by one short facade) even if the OSM stage calls
    it reliable."""
    lines = json.loads(paths.roi_json.read_text())["lines"]
    frame = LocalFrame(*lines[0][0])
    lines_xy = [[frame.xy(*p) for p in line] for line in lines]
    worst = {}
    for _, cam in cameras.values():
        pipe = cam.get("pipeline")
        if pipe and cam["reliable"]:
            d = min(distance_to_polyline(frame.xy(pipe["lon"], pipe["lat"]), line) for line in lines_xy)
            worst[cam["leg"]] = max(worst.get(cam["leg"], 0.0), d)
    return {leg: d for leg, d in worst.items() if d > max_off_street_m}


def _apply_alignment(paths, mapillary, aligned_only=False, max_off_street_m=15.0):
    """Put the reliable aligned poses into the Mapillary metadata; returns the summary."""
    images, counts = {}, {"overridden": 0, "added": 0, "kept_mapillary": 0, "aspect_mismatch": 0}
    cameras = _aligned_cameras(paths)
    off_street = _off_street_legs(paths, cameras, max_off_street_m)
    for leg, d in off_street.items():
        print(f"  align_poses: {leg} rejected: its aligned cameras end up {d:.0f} m from the street")
    for img_id, (folder, cam) in cameras.items():
        pipe = cam.get("pipeline")
        info = {"sequence_folder": folder, "file": cam["file"], "leg": cam.get("leg"), "reliable": cam["reliable"],
                "leg_fit_to_walls_m": cam.get("leg_facade_to_wall_median_m")}
        if cam.get("leg") in off_street:
            info["reason"] = (f"{cam['leg']} rejected: its aligned cameras end up {off_street[cam['leg']]:.0f} m "
                              f"from the street (more than {max_off_street_m:.0f} m)")
            if img_id in mapillary:
                images[img_id] = {"source": "mapillary", **info}
            continue
        if not (pipe and cam["reliable"]):
            info["reason"] = cam.get("not_aligned_reason") or "leg not reliable (fits the OSM walls worse than 2 m)"
            if img_id in mapillary:
                images[img_id] = {"source": "mapillary", **info}
            continue

        R_wc = np.array(pipe["rotation_cam_to_enu"]).T
        rotvec = Rotation.from_matrix(R_wc).as_rotvec()
        yaw, pitch, roll = Rotation.from_rotvec(rotvec).as_euler("ZYX", degrees=True)
        intr = pipe["intrinsics"]
        pose = {"latitude": pipe["lat"], "longitude": pipe["lon"], "rotvec": rotvec.tolist(),
                "yaw": round(float(yaw), 2), "pitch": round(float(pitch), 2), "roll": round(float(roll), 2),
                "camera_height_m": pipe["height_m"], "pose_source": "colmap"}
        info["camera_height_m"] = round(pipe["height_m"], 3)

        meta = mapillary.get(img_id)
        if meta is None:
            mapillary[img_id] = {**pose, "focal_length_pixels": intr["fx"], "original_width": intr["width"],
                                 "original_height": intr["height"],
                                 "image_url": str(paths.images / folder / cam["file"])}
            images[img_id] = {"source": "colmap (added)", **info}
            counts["added"] += 1
            continue
        w, h = meta.get("original_width"), meta.get("original_height")
        if not (w and h) or abs(w / h - intr["width"] / intr["height"]) > 0.02:
            images[img_id] = {"source": "mapillary", **info, "reason": "image size differs from the aligned download"}
            counts["aspect_mismatch"] += 1
            continue
        frame = LocalFrame(meta["longitude"], meta["latitude"])
        info["moved_m"] = round(math.dist((0.0, 0.0), frame.xy(pipe["lon"], pipe["lat"])), 3)
        if meta.get("rotvec") is not None:
            old = _heading(Rotation.from_rotvec(meta["rotvec"]).as_matrix())
            info["heading_change_deg"] = round(wrap_deg(_heading(R_wc) - old), 2)
        meta.update(pose, focal_length_pixels=intr["fx"] * w / intr["width"])
        images[img_id] = {"source": "colmap", **info}
        counts["overridden"] += 1

    for img_id in mapillary:
        if img_id not in images:
            images[img_id] = {"source": "mapillary", "reason": "not in an aligned capture"}
    counts["kept_mapillary"] = sum(v["source"] == "mapillary" for v in images.values())

    used = counts["overridden"] + counts["added"]
    if aligned_only and used:
        for img_id in [k for k, v in mapillary.items() if v.get("pose_source") != "colmap"]:
            del mapillary[img_id]
            images[img_id]["source"] = "dropped (--aligned-only)"
        counts["kept_mapillary"] = 0
    elif aligned_only:
        print("  align_poses: no reliable aligned image, so --aligned-only is ignored")

    summary = {"pose_source": "colmap", "roi_folder": str(paths.root), "aligned_only": aligned_only,
               "rejected_legs_off_street_m": {leg: round(d, 1) for leg, d in off_street.items()}, **counts}
    (paths.root / "pose_overrides.json").write_text(json.dumps({**summary, "images": images}, indent=1))
    print(f"  align_poses: {counts['overridden']} images got aligned poses, {counts['added']} aligned images "
          f"added, {counts['kept_mapillary']} kept Mapillary poses"
          + (f", {len(images) - used - counts['kept_mapillary']} dropped" if aligned_only and used else "")
          + f" (details: {paths.root / 'pose_overrides.json'})")
    return summary


def align_poses(value, state):
    source = state.get_metadata("pose_source", "raw")
    mapillary = state.require_metadata("provider_data")["mapillary"]
    if source == "computed":
        _use_computed(mapillary)
    elif source == "colmap":
        try:
            paths = _run_alignment(state.require_metadata("bbox"), state.get_metadata("alignment_dir", None))
            state.set_metadata("pose_alignment", _apply_alignment(paths, mapillary,
                                                                  state.get_metadata("aligned_only", False)))
        except (Exception, SystemExit) as e:
            print(f"  align_poses: alignment failed ({e}); every image keeps its Mapillary pose")
    return value
