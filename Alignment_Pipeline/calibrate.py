"""Stage 4 (calibrate): camera calibration and gravity with GeoCalib (cvg/GeoCalib, ECCV 2024).

Per sequence folder of the ROI:
  1. Calibration: GeoCalib's "distorted" model (simple radial camera) on --sample frames spread
     over the sequence; the median focal and k1 are the sequence's calibration (one camera).
  2. Gravity: every frame again, with the focal fixed to that median as a prior, for its roll,
     pitch and up direction in the camera frame.
Frames are shrunk to --work-size on the CPU first (GeoCalib works at 320 px internally; focals
are kept as fractions of the image's long side, so the scale does not matter).
Writes data/calibration/<sequence folder>.json:
  focal_norm, focal_px, k1, width, height       the calibration (GeoCalib simple radial; reconstruct.camera_setup
                                                picks the COLMAP model)
  frames: {file: {roll_deg, pitch_deg, up_cam}} up_cam: unit up vector in the camera frame
                                                (x right, y down, z forward); gravity = -up_cam

Usage: python calibrate.py rois/mapillary_roi_2 [--sample 20] [--force]
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from common import RoiPaths

warnings.filterwarnings("ignore", category=FutureWarning)
from geocalib import GeoCalib  # noqa: E402


def load(model, path, work_size):
    im = model.load_image(path)
    s = work_size / max(im.shape[-2:])
    if s < 1:
        im = torch.nn.functional.interpolate(im[None], scale_factor=s, mode="area", recompute_scale_factor=False)[0]
    return im


def calibrate_sequence(images_dir, files, out_json, sample=20, work_size=640, device="mps"):
    t0 = time.time()
    dev = torch.device(device)
    model = GeoCalib(weights="distorted").to(dev)
    imgs = [load(model, images_dir / f, work_size) for f in files]
    L = max(imgs[0].shape[-2:])
    from PIL import Image
    w, h = Image.open(images_dir / files[0]).size

    idx = np.unique(np.linspace(0, len(files) - 1, min(sample, len(files))).round().astype(int))
    fs, ks = [], []
    for i in idx:
        r = model.calibrate(imgs[i].to(dev), camera_model="simple_radial")
        fs.append(float(r["camera"].f[0, 0].cpu()) / L)
        ks.append(float(r["camera"].k1[0].cpu()))
    f_norm, k1 = float(np.median(fs)), float(np.median(ks))
    t_cal = time.time() - t0

    frames = {}
    prior = torch.tensor(f_norm * L, device=dev)
    for f, im in zip(files, imgs):
        r = model.calibrate(im.to(dev), camera_model="simple_radial", priors={"focal": prior})
        g = r["gravity"]
        roll, pitch = (float(np.degrees(v)) for v in g.rp[0].cpu())
        frames[f] = {"roll_deg": round(roll, 3), "pitch_deg": round(pitch, 3),
                     "up_cam": [round(float(v), 6) for v in g.vec3d[0].cpu()]}
    doc = {"focal_norm": f_norm, "focal_px": f_norm * max(w, h), "k1": k1, "width": w, "height": h,
           "camera_model": "SIMPLE_RADIAL", "colmap_params": [f_norm * max(w, h), w / 2, h / 2, k1],
           "sample_frames": [files[i] for i in idx], "sample_focal_norm": fs, "sample_k1": ks,
           "focal_spread_mad": float(np.median(np.abs(np.array(fs) - f_norm))),
           "seconds": {"calibration": round(t_cal, 1), "gravity": round(time.time() - t0 - t_cal, 1)},
           "frames": frames}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1))
    return doc


def calibrate_roi(roi_dir, sample=20, work_size=640, force=False, device=None):
    p = RoiPaths(roi_dir)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    for s in p.sequences():
        folder = s["folder"]
        out = p.calibration_json(folder)
        files = sorted(p.frames(folder))
        if out.exists() and not force and set(json.loads(out.read_text())["frames"]) == set(files):
            print(f"{folder}: GeoCalib calibration exists ({out})")
            continue
        d = calibrate_sequence(p.images / folder, files, out, sample, work_size, device)
        mly = [f.get("camera_parameters") for f in p.frames(folder).values() if f.get("camera_parameters")]
        mly_f = sorted({round(m[0], 3) for m in mly})
        print(f"{folder}: focal {d['focal_norm']:.3f} of long side ({d['focal_px']:.0f} px, spread "
              f"{d['focal_spread_mad']:.3f}), k1 {d['k1']:+.3f}; gravity for {len(files)} frames "
              f"(roll median {np.median([v['roll_deg'] for v in d['frames'].values()]):+.1f} deg); "
              f"{d['seconds']['calibration'] + d['seconds']['gravity']:.0f} s. Mapillary focals: {mly_f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("--sample", type=int, default=20, help="Frames used for the focal and k1")
    parser.add_argument("--work-size", type=int, default=640)
    parser.add_argument("--device", help="mps or cpu (default: mps if available)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    calibrate_roi(args.roi_dir, args.sample, args.work_size, args.force, args.device)


if __name__ == "__main__":
    main()
