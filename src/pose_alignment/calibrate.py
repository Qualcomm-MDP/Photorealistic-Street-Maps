"""Stage 4 (calibrate): camera calibration and gravity with GeoCalib (cvg/GeoCalib, ECCV 2024).

Per sequence folder of the ROI:
  1. Calibration: GeoCalib's "distorted" model (simple radial camera) on --sample frames (10) spread
     over the sequence; the median focal and k1 are the sequence's calibration (one camera).
  2. Gravity: every --gravity-every-th frame (default 5) and the last, with the focal fixed to
     that median, for its roll, pitch and up direction in the camera frame. Each estimate starts
     from the previous one with at most 10 optimiser steps (GeoCalib's own "gravity prior" would
     fix the gravity, not start from it, so the start is set in its optimiser instead). Where two
     estimates differ by more than --gravity-refill degrees (default 4), the frames between are
     estimated too; the rest are interpolated. --gravity-every 1 estimates every frame from a cold
     start (the old way). Against the COLMAP models' rotations (roi_6 motorcycle, roi_19, 24, 27),
     every 5th frame was 3-5x faster and its gravity 0.04-0.15 deg further off (median) than every
     frame's (0.44-0.48 deg on the cars, 1.27 on the motorcycle); every 3rd was no closer. A 2 deg
     refill fired on GeoCalib's own frame-to-frame scatter rather than real changes.
With --calibration mapillary, step 1 is Mapillary's focal, k1 and k2 instead (the sequence's
most common camera_parameters; GeoCalib's where Mapillary has none), written to
data/calibration_mapillary/; step 2 is the same, with Mapillary's focal as the prior.
Crops of a 360 camera (mapillary_layer.py) are pinhole views with an exactly known focal and no
distortion: step 1 is that focal (source "pano") with either --calibration; step 2 the same.

A warning suggests --calibration mapillary when GeoCalib's calibration looks unreliable: its
optimiser did not converge on most of the sampled frames, or cameras of one rig (same make,
model and image size, so the same lens) got focals more than 15% apart. Both happen with lenses
wider than GeoCalib's model holds (roi_21's 2017 Roush rig: 0.42-0.65 for six identical cameras,
Mapillary 0.48).
Frames are shrunk to --work-size on the CPU first (GeoCalib works at 320 px internally; focals
are kept as fractions of the image's long side, so the scale does not matter).
Writes data/calibration/<sequence folder>.json:
  focal_norm, focal_px, k1, width, height       the calibration (GeoCalib simple radial; reconstruct.camera_setup
                                                picks the COLMAP model); k2 too from Mapillary
  source                                        geocalib or mapillary
  sample_not_converged, gravity_not_converged   frames where GeoCalib's optimiser hit its step limit
  frames: {file: {roll_deg, pitch_deg, up_cam,  up_cam: unit up vector in the camera frame
                  estimated}}                   (x right, y down, z forward); gravity = -up_cam;
                                                estimated false: interpolated

Against COLMAP's refined focal (12 normal-lens sequences), the median of 10 frames was 5.4% off
(median), 20 frames 6.0%, 5 frames 5.0%, 1 frame 16%: past about 5 frames GeoCalib's own bias, not the
sample, sets the error (logs/geocalib_focal_samples.png).

Usage: python -m pose_alignment.calibrate outputs/alignment/mapillary_roi_2 [--sample 10] [--force] [--calibration mapillary]
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from .geo_common import RoiPaths, mapillary_calibration

warnings.filterwarnings("ignore", category=FutureWarning)
import geocalib.lm_optimizer as _lmo  # noqa: E402
from geocalib import GeoCalib  # noqa: E402
from geocalib.gravity import Gravity  # noqa: E402

RIG_SPREAD = 0.15  # rig cameras' GeoCalib focals further apart than this: unreliable
UNRELIABLE = (
    "GeoCalib calibration looks unreliable for this camera; try --calibration mapillary"
)


class NotConverged(logging.Handler):
    """Counts GeoCalib's "Reached maximum number of steps without convergence" warnings and keeps
    them off the console (the stage reports the counts)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += "without convergence" in record.getMessage()


NOT_CONVERGED = NotConverged()
_lm_log = logging.getLogger("geocalib.lm_optimizer")
_lm_log.addHandler(NOT_CONVERGED)
_lm_log.propagate = False


# GeoCalib's "gravity" prior fixes the gravity rather than starting from it, so a warm start is
# injected into the optimiser's initial estimate instead (the up vector to start from, or None)
_WARM = {"up": None}
_trivial_estimation = _lmo.get_trivial_estimation


def _warm_estimation(data, camera_model):
    camera, gravity = _trivial_estimation(data, camera_model)
    if _WARM["up"] is not None and "prior_gravity" not in data:
        gravity = Gravity(
            torch.as_tensor(
                _WARM["up"], dtype=torch.float32, device=gravity.vec3d.device
            )[None]
        )
    return camera, gravity


_lmo.get_trivial_estimation = _warm_estimation


def converged(model, im, **kw):
    """(GeoCalib result, whether its optimiser converged)."""
    before = NOT_CONVERGED.count
    r = model.calibrate(im, camera_model="simple_radial", **kw)
    return r, NOT_CONVERGED.count == before


def rp_from_up(up):
    """(roll, pitch) in degrees of an up vector in the camera frame (GeoCalib's convention)."""
    return float(np.degrees(np.arctan2(-up[0], -up[1]))), float(
        np.degrees(np.arcsin(np.clip(up[2], -1, 1)))
    )


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def gravity_frames(
    model, imgs, files, prior, dev, every=5, refill_deg=4.0, warm_steps=10
):
    """Per-frame gravity: GeoCalib on every `every`-th frame in capture order (and the last), each
    starting from the previous estimate with at most warm_steps optimiser steps; where two
    estimates differ by more than refill_deg, the frames between are estimated too; the rest are
    interpolated. Returns ({file: {roll_deg, pitch_deg, up_cam, estimated}}, estimates that did not
    converge, number of estimates). every=1, warm_steps=None: every frame from a cold start (the
    old behaviour)."""
    opt = model.model.optimizer
    cold_steps = opt.num_steps
    up, bad = {}, 0

    def estimate(i, start):
        nonlocal bad
        _WARM["up"] = start if warm_steps else None
        opt.num_steps = warm_steps if (warm_steps and start is not None) else cold_steps
        try:
            r, ok = converged(model, imgs[i].to(dev), priors={"focal": prior})
        finally:
            _WARM["up"], opt.num_steps = None, cold_steps
        bad += not ok
        up[i] = r["gravity"].vec3d[0].cpu().numpy().astype(float)

    n = len(files)
    keys = sorted(set(range(0, n, max(1, every))) | {n - 1})
    for i in keys:
        estimate(i, up[max(up)] if up else None)
    for a, b in zip(keys, keys[1:]):
        if b - a > 1 and angle_deg(up[a], up[b]) > refill_deg:
            for i in range(a + 1, b):
                estimate(i, up[i - 1])
    est = sorted(up)
    for i in range(n):
        if i not in up:  # between two estimates: normalised linear interpolation
            a = max(k for k in est if k < i)
            b = min(k for k in est if k > i)
            v = up[a] + (up[b] - up[a]) * (i - a) / (b - a)
            up[i] = v / np.linalg.norm(v)
    frames = {}
    for i, f in enumerate(files):
        roll, pitch = rp_from_up(up[i])
        frames[f] = {
            "roll_deg": round(roll, 3),
            "pitch_deg": round(pitch, 3),
            "up_cam": [round(float(v), 6) for v in up[i]],
            "estimated": i in est,
        }
    return frames, bad, len(est)


def load(model, path, work_size):
    im = model.load_image(path)
    s = work_size / max(im.shape[-2:])
    if s < 1:
        im = torch.nn.functional.interpolate(
            im[None], scale_factor=s, mode="area", recompute_scale_factor=False
        )[0]
    return im


def calibrate_sequence(
    images_dir,
    files,
    out_json,
    sample=10,
    work_size=640,
    device="mps",
    fixed=None,
    source="geocalib",
    gravity=None,
):
    """GeoCalib's calibration (or fixed = a known [focal, k1, k2]: Mapillary's, or a 360 crop's) and
    per-frame gravity."""
    t0 = time.time()
    dev = torch.device(device)
    model = GeoCalib(weights="distorted").to(dev)
    imgs = [load(model, images_dir / f, work_size) for f in files]
    L = max(imgs[0].shape[-2:])
    from PIL import Image

    w, h = Image.open(images_dir / files[0]).size

    idx = np.unique(
        np.linspace(0, len(files) - 1, min(sample, len(files))).round().astype(int)
    )
    fs, ks, sample_bad = [], [], 0
    if fixed:
        f_norm, k1, k2 = fixed
        idx = []
    for i in idx:
        r, ok = converged(model, imgs[i].to(dev))
        sample_bad += not ok
        fs.append(float(r["camera"].f[0, 0].cpu()) / L)
        ks.append(float(r["camera"].k1[0].cpu()))
    if not fixed:
        f_norm, k1, k2 = float(np.median(fs)), float(np.median(ks)), None
    t_cal = time.time() - t0

    prior = torch.tensor(f_norm * L, device=dev)
    frames, gravity_bad, n_est = gravity_frames(
        model, imgs, files, prior, dev, **(gravity or {})
    )
    doc = {
        "source": source if fixed else "geocalib",
        "focal_norm": f_norm,
        "focal_px": f_norm * max(w, h),
        "k1": k1,
        "width": w,
        "height": h,
    }
    if fixed:
        doc.update(
            k2=k2,
            camera_model="RADIAL",
            colmap_params=[f_norm * max(w, h), w / 2, h / 2, k1, k2],
        )
    else:
        doc.update(
            camera_model="SIMPLE_RADIAL",
            colmap_params=[f_norm * max(w, h), w / 2, h / 2, k1],
            sample_frames=[files[i] for i in idx],
            sample_focal_norm=fs,
            sample_k1=ks,
            focal_spread_mad=float(np.median(np.abs(np.array(fs) - f_norm))),
            sample_not_converged=sample_bad,
        )
    doc.update(
        gravity_not_converged=gravity_bad,
        gravity_estimated=n_est,
        gravity_settings=gravity or {},
        frames=frames,
        seconds={
            "calibration": round(t_cal, 1),
            "gravity": round(time.time() - t0 - t_cal, 1),
        },
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1))
    return doc


def calibrate_roi(
    roi_dir,
    sample=10,
    work_size=640,
    force=False,
    device=None,
    calibration="geocalib",
    gravity_every=5,
    gravity_refill=4.0,
):
    p = RoiPaths(roi_dir, calibration=calibration)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    docs = {}
    for s in p.sequences():
        folder = s["folder"]
        out = p.calibration_json(folder)
        meta = p.frames(folder)
        files = sorted(meta)
        if (
            out.exists()
            and not force
            and set(json.loads(out.read_text())["frames"]) == set(files)
        ):
            print(f"{folder}: {calibration} calibration exists ({out})")
            docs[folder] = json.loads(out.read_text())
            continue
        _, mly = mapillary_calibration(p, folder)
        pano = p.pano(folder)
        if pano:  # a 360 crop: exact pinhole (camera_parameters [focal, 0, 0])
            fixed, source = mly, "pano"
        else:
            if calibration == "mapillary" and not mly:
                print(
                    f"{folder}: no Mapillary calibration for this sequence; using GeoCalib's"
                )
            fixed, source = (
                (mly, "mapillary") if calibration == "mapillary" else (None, "geocalib")
            )
        d = calibrate_sequence(
            p.images / folder,
            files,
            out,
            sample,
            work_size,
            device,
            fixed,
            source,
            {
                "every": gravity_every,
                "refill_deg": gravity_refill,
                "warm_steps": 10 if gravity_every > 1 else None,
            },
        )  # 1: the old every-frame cold start
        docs[folder] = d
        mly_f = sorted(
            {
                round(f["camera_parameters"][0], 3)
                for f in meta.values()
                if f.get("camera_parameters")
            }
        )
        what = (
            f"360 {pano['side']} crop, exact focal {d['focal_norm']:.3f} of the side ({d['focal_px']:.0f} px, "
            f"{pano['fov_deg']:g} deg), no distortion"
            if d["source"] == "pano"
            else f"Mapillary focal {d['focal_norm']:.3f} of long side ({d['focal_px']:.0f} px), k1 {d['k1']:+.3f}, "
            f"k2 {d['k2']:+.3f}"
            if d["source"] == "mapillary"
            else f"focal {d['focal_norm']:.3f} of long side ({d['focal_px']:.0f} px, spread {d['focal_spread_mad']:.3f}), "
            f"k1 {d['k1']:+.3f}, optimiser did not converge on {d['sample_not_converged']}/"
            f"{len(d['sample_frames'])} sampled frames"
        )
        print(
            f"{folder}: {what}; gravity for {len(files)} frames ({d['gravity_estimated']} estimated, roll median "
            f"{np.median([v['roll_deg'] for v in d['frames'].values()]):+.1f} deg, did not converge on "
            f"{d['gravity_not_converged']}); {d['seconds']['calibration'] + d['seconds']['gravity']:.0f} s. "
            f"Mapillary focals: {mly_f}"
        )
    if calibration == "geocalib":
        warn_unreliable(p, docs)


def warn_unreliable(p, docs):
    """Print UNRELIABLE for each sequence whose GeoCalib optimiser did not converge on most sampled
    frames, or whose rig siblings (same make, model and image size) got focals more than
    RIG_SPREAD apart."""
    reasons = {}
    for folder, d in docs.items():
        n = len(d.get("sample_frames", []))
        if (
            d.get("source", "geocalib") == "geocalib"
            and n
            and d.get("sample_not_converged", 0) > n / 2
        ):
            reasons.setdefault(folder, []).append(
                f"no convergence on {d['sample_not_converged']}/{n} sampled frames"
            )
    rigs = {}
    for folder, d in docs.items():
        f0 = next(iter(p.frames(folder).values()))
        rigs.setdefault(
            (f0.get("make"), f0.get("model"), d["width"], d["height"]), []
        ).append(folder)
    for (make, model, *_), folders in rigs.items():
        focals = [docs[f]["focal_norm"] for f in folders]
        if len(folders) > 1 and max(focals) / min(focals) - 1 > RIG_SPREAD:
            for f in folders:
                reasons.setdefault(f, []).append(
                    f"{len(folders)} {make} {model} rig cameras got focals "
                    f"{min(focals):.2f}-{max(focals):.2f}"
                )
    for folder, why in reasons.items():
        print(f"{folder}: WARNING: {UNRELIABLE} ({'; '.join(why)})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument(
        "--sample", type=int, default=10, help="Frames used for the focal and k1"
    )
    parser.add_argument("--work-size", type=int, default=640)
    parser.add_argument("--device", help="mps or cpu (default: mps if available)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--calibration",
        choices=["geocalib", "mapillary"],
        default="geocalib",
        help="Focal and distortion from GeoCalib (default) or Mapillary; gravity from GeoCalib either way",
    )
    parser.add_argument(
        "--gravity-every",
        type=int,
        default=5,
        help="Estimate gravity on every n-th frame and interpolate between (1: every frame)",
    )
    parser.add_argument(
        "--gravity-refill",
        type=float,
        default=4.0,
        help="Also estimate the frames between two estimates further apart than this (deg)",
    )
    args = parser.parse_args()
    calibrate_roi(
        args.roi_dir,
        args.sample,
        args.work_size,
        args.force,
        args.device,
        args.calibration,
        args.gravity_every,
        args.gravity_refill,
    )


if __name__ == "__main__":
    main()
