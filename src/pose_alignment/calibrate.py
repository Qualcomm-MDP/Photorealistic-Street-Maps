"""Stage 4 (calibrate): each sequence's camera calibration and per-frame gravity, from Mapillary.

Per sequence folder of the ROI, from its frames' cameras.json (data_ingest/mapillary_sequences.py):
  1. Calibration: the sequence's most common Mapillary camera_parameters: focal (a fraction of the
     image's long side), k1 and k2, from Mapillary's own structure from motion. Mapillary's focal
     is sometimes far off (roi_15's iPhone: 1.338 against about 0.8; a Google Pixel 3 in downtown
     Ann Arbor: 0.976, 0.918 and 0.732), so each sequence is checked against the other sequences
     of the same camera in the ROI (make, model and image size): when at least two others have a
     calibration and this one's focal is more than FOCAL_TOLERANCE (15%) from their median, it
     takes the calibration of the one closest to that median instead (source "mapillary_median",
     "correction" says why). With only two sequences of a camera there is no majority: both are
     kept, with a warning when they differ by more than 15%. A sequence with no Mapillary
     calibration starts from COLMAP's usual
     guess, DEFAULT_FOCAL (1.2) x the long side with no distortion (source "default"). Bundle
     adjustment refines the focal and distortion either way (reconstruct.py).
     Crops of a 360 camera are pinhole views with an exactly known focal and no distortion
     (source "pano"); reconstruct keeps them fixed.
  2. Gravity: each frame's up vector in the camera frame, from its computed_rotation (OpenSfM,
     world-to-camera in east/north/up): that rotation applied to the world's up. Frames without
     one are interpolated between their neighbours. A sequence with no rotation at all gets no
     gravity ("frames" empty, gravity_source null): reconstruct solves it without gravity priors
     and align levels it from its road points.
Writes data/calibration/<sequence folder>.json:
  source            mapillary, mapillary_median, default or pano
  correction        why the focal was replaced (mapillary_median), else null
  focal_norm, focal_px, k1, k2, width, height, camera_model (RADIAL), colmap_params
  gravity_source    "mapillary", or null when the sequence has no rotations
  gravity_estimated frames with their own rotation
  frames: {file: {roll_deg, pitch_deg, up_cam,  up_cam: unit up vector in the camera frame
                  estimated}}                   (x right, y down, z forward); gravity = -up_cam;
                                                estimated false: interpolated

Earlier versions of this stage estimated the focal, distortion and gravity from the photos
themselves with GeoCalib (cvg/GeoCalib). Their calibrations (source "geocalib", or no
gravity_source) are recalculated when found.

Usage: python -m pose_alignment.calibrate outputs/alignment/mapillary_roi_2 [--force]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geo_common import RoiPaths, mapillary_calibration

SOURCES = ("mapillary", "mapillary_median", "default", "pano")
FOCAL_TOLERANCE = (
    0.15  # a focal further than this from its camera's other sequences is replaced
)
DEFAULT_FOCAL = 1.2  # COLMAP's starting focal without a calibration, x the long side


def rp_from_up(up):
    """(roll, pitch) in degrees of an up vector in the camera frame (x right, y down, z forward)."""
    return float(np.degrees(np.arctan2(-up[0], -up[1]))), float(
        np.degrees(np.arcsin(np.clip(up[2], -1, 1)))
    )


def up_from_rotation(rotvec):
    """A camera's up vector in its own frame (x right, y down, z forward) from a Mapillary
    computed_rotation: OpenSfM's world-to-camera rotation (axis-angle), world east/north/up."""
    return Rotation.from_rotvec(rotvec).as_matrix() @ np.array([0.0, 0.0, 1.0])


def mapillary_gravity(meta, files):
    """Per-frame gravity from Mapillary's computed rotations, or None when no frame has one.
    files: in capture order; a frame without a rotation is interpolated between its nearest
    neighbours that have one (estimated false), or takes the nearest one at either end. Returns
    ({file: {roll_deg, pitch_deg, up_cam, estimated}}, number of frames with their own rotation)."""
    up = {
        i: up_from_rotation(meta[f]["computed_rotation"])
        for i, f in enumerate(files)
        if meta[f].get("computed_rotation")
    }
    if not up:
        return None
    known = sorted(up)
    frames = {}
    for i, f in enumerate(files):
        if i in up:
            v = up[i]
        else:
            before = [k for k in known if k < i]
            after = [k for k in known if k > i]
            if before and after:
                a, b = before[-1], after[0]
                v = up[a] + (up[b] - up[a]) * (i - a) / (b - a)
            else:
                v = up[(before or after)[-1 if before else 0]]
        v = v / np.linalg.norm(v)
        roll, pitch = rp_from_up(v)
        frames[f] = {
            "roll_deg": round(roll, 3),
            "pitch_deg": round(pitch, 3),
            "up_cam": [round(float(x), 6) for x in v],
            "estimated": i in up,
        }
    return frames, len(known)


def is_current(doc):
    """Whether a calibration JSON comes from this version of the stage (not GeoCalib's)."""
    return doc.get("source") in SOURCES and "gravity_source" in doc


def camera_key(meta):
    """A sequence's camera: (make, model, width, height) of its first frame."""
    f0 = next(iter(meta.values()))
    return (f0.get("make"), f0.get("model"), f0.get("width"), f0.get("height"))


def choose_calibrations(mly, cameras, tolerance=FOCAL_TOLERANCE):
    """Each sequence's calibration from Mapillary's, checked against its camera's other sequences.

    mly: {folder: [focal, k1, k2] or None}; cameras: {folder: camera_key}. Returns
    {folder: (source, [focal, k1, k2], correction)}: Mapillary's, or with at least two other
    sequences of the same camera calibrated and this focal more than tolerance from their median,
    the calibration of the one closest to that median; DEFAULT_FOCAL without any."""
    out = {}
    for folder, own in mly.items():
        if not own:
            out[folder] = ("default", [DEFAULT_FOCAL, 0.0, 0.0], None)
            continue
        others = [
            mly[f]
            for f in mly
            if f != folder and mly[f] and cameras[f] == cameras[folder]
        ]
        if len(others) >= 2:
            median = float(np.median([c[0] for c in others]))
            if abs(own[0] / median - 1) > tolerance:
                closest = min(others, key=lambda c: abs(c[0] - median))
                out[folder] = (
                    "mapillary_median",
                    list(closest),
                    f"Mapillary's focal {own[0]:.3f} is {100 * (own[0] / median - 1):+.0f}% from "
                    f"the median {median:.3f} of {len(others)} other sequences of this camera",
                )
                continue
        out[folder] = ("mapillary", list(own), None)
    return out


def focal_warnings(mly, cameras, tolerance=FOCAL_TOLERANCE):
    """Warnings for cameras with exactly two calibrated sequences whose focals differ by more than
    tolerance: without a third there is no telling which is right, so both are kept."""
    groups = {}
    for folder, own in mly.items():
        if own:
            groups.setdefault(cameras[folder], []).append((folder, own[0]))
    out = []
    for (make, model, *_), seqs in groups.items():
        if len(seqs) == 2:
            (a, fa), (b, fb) = seqs
            if abs(fa / fb - 1) > tolerance:
                out.append(
                    f"WARNING: the {make} {model} sequences {a} and {b} have Mapillary focals "
                    f"{fa:.3f} and {fb:.3f}, {100 * abs(fa / fb - 1):.0f}% apart; kept both "
                    "(no third to compare with): check the reconstruction (shape_check.py)"
                )
    return out


def calibrate_sequence(
    images_dir, files, out_json, source, calibration, meta, correction=None
):
    """Write one sequence's calibration: calibration = [focal (fraction of the long side), k1, k2]
    and source as chosen by choose_calibrations (or "pano"), gravity from meta (the frames'
    cameras.json records)."""
    from PIL import Image

    w, h = Image.open(images_dir / files[0]).size
    f_norm, k1, k2 = (float(x) for x in calibration)
    gravity = mapillary_gravity(meta, files)
    frames, n_own = gravity if gravity else ({}, 0)
    doc = {
        "source": source,
        "correction": correction,
        "focal_norm": f_norm,
        "focal_px": f_norm * max(w, h),
        "k1": k1,
        "k2": k2,
        "width": w,
        "height": h,
        "camera_model": "RADIAL",
        "colmap_params": [f_norm * max(w, h), w / 2, h / 2, k1, k2],
        "gravity_source": "mapillary" if gravity else None,
        "gravity_estimated": n_own,
        "frames": frames,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1))
    return doc


def calibrate_roi(roi_dir, force=False):
    p = RoiPaths(roi_dir)
    sequences = [s["folder"] for s in p.sequences()]
    metas = {folder: p.frames(folder) for folder in sequences}
    panos = {folder: p.pano(folder) for folder in sequences}
    mly = {folder: mapillary_calibration(p, folder)[1] for folder in sequences}
    checked = {f: mly[f] for f in sequences if not panos[f]}
    cameras = {f: camera_key(metas[f]) for f in checked}
    chosen = choose_calibrations(checked, cameras)
    for warning in focal_warnings(checked, cameras):
        print(warning)
    docs = {}
    for folder in sequences:
        out = p.calibration_json(folder)
        meta = metas[folder]
        files = sorted(meta)
        if out.exists() and not force:
            old = json.loads(out.read_text())
            if not is_current(old):
                print(
                    f"{folder}: {out} is from the GeoCalib version of this stage; recalculating"
                )
            elif set(old["frames"]) == set(files) or not old["gravity_source"]:
                print(f"{folder}: calibration exists ({out})")
                docs[folder] = old
                continue
        pano = panos[folder]
        if pano:  # a 360 crop: an exact pinhole (camera_parameters [focal, 0, 0])
            source, calib, correction = "pano", mly[folder], None
        else:
            source, calib, correction = chosen[folder]
        d = calibrate_sequence(
            p.images / folder, files, out, source, calib, meta, correction
        )
        docs[folder] = d
        what = (
            f"360 {pano['side']} crop, exact focal {d['focal_norm']:.3f} of the side "
            f"({d['focal_px']:.0f} px, {pano['fov_deg']:g} deg), no distortion"
            if source == "pano"
            else f"Mapillary focal {d['focal_norm']:.3f} of the long side ({d['focal_px']:.0f} px), "
            f"k1 {d['k1']:+.3f}, k2 {d['k2']:+.3f}"
            + (f" (replaced: {correction})" if correction else "")
            if source != "default"
            else f"WARNING: no Mapillary calibration; starting from {DEFAULT_FOCAL} x the long "
            f"side ({d['focal_px']:.0f} px), no distortion, for bundle adjustment to refine"
        )
        if d["gravity_source"]:
            roll = np.median([v["roll_deg"] for v in d["frames"].values()])
            how = (
                f"gravity for {len(files)} frames ({d['gravity_estimated']} with their own "
                f"rotation, roll median {roll:+.1f} deg)"
            )
        else:
            how = (
                "WARNING: no Mapillary rotations, so no gravity: reconstructed without gravity "
                "priors and levelled from the road"
            )
        print(f"{folder}: {what}; {how}")
    return docs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    calibrate_roi(args.roi_dir, args.force)


if __name__ == "__main__":
    main()
