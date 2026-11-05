"""Tool: how consistent Mapillary's per-frame gravity is with the reconstructed cameras.

A street has one "up". Turned into a reconstruction's frame by each camera's COLMAP rotation,
a good per-frame gravity points the same way for every camera of a leg; the spread around their
robust mean (align.level_with_gravity, the levelling the align stage uses) measures how far off
the per-frame values are. No ground truth needed. The gravity is the calibration's (Mapillary's
computed rotations, calibrate.py), read straight from cameras.json when the calibration has none.

The reconstruction used that gravity for its priors, which pulls its camera rotations towards it,
so a small spread is a lower bound: a bias the same on every frame shows only where the camera
turns. The alignment's fit to OSM (summary.json) is the final judge.

Usage: python -m pose_alignment.gravity_check outputs/alignment/mapillary_roi_19 [--tag t]
Writes results/[tag/]gravity_check.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .align import level_with_gravity
from .calibrate import mapillary_gravity
from .geo_common import RoiPaths, load_cameras


def gravity_of(p, folder):
    """{"frames": {file: {"up_cam": ...}}} for one sequence: the calibration's gravity, else
    Mapillary's rotations from cameras.json; None without either."""
    f = p.calibration_json(folder)
    if f.exists():
        gc = json.loads(f.read_text())
        if gc.get("frames"):
            return gc
    meta = p.frames(folder)
    mly = mapillary_gravity(meta, sorted(meta))
    return {"frames": mly[0]} if mly else None


def spread(names, C, R, gc):
    """(median deg from the mean up, outlier cameras, cameras used) on the cameras gc has."""
    have = [i for i, n in enumerate(names) if n in gc["frames"]]
    if len(have) < 3:
        return None
    _, _, med, out = level_with_gravity([names[i] for i in have], C[have], R[have], gc)
    return med, out, len(have)


def check(roi_dir, tag=None):
    p = RoiPaths(roi_dir, tag)
    if not p.legs_json.exists():
        raise SystemExit(
            f"No reconstruction in {p.reconstruction}: run reconstruct first"
        )
    doc = json.loads(p.legs_json.read_text())
    rows = []
    for leg in doc.get("legs", []):
        for cam in leg["cameras"]:
            if "model" not in cam:
                continue
            names, C, R = load_cameras(p.reconstruction / cam["model"])
            gc = gravity_of(p, cam["folder"])
            r = spread(list(names), np.asarray(C), np.asarray(R), gc) if gc else None
            rows.append(
                {
                    "leg": leg["id"],
                    "folder": cam["folder"],
                    "spread_median_deg": None if r is None else round(r[0], 3),
                    "outlier_cameras": None if r is None else r[1],
                    "cameras": len(names) if r is None else r[2],
                }
            )
    if not rows:
        raise SystemExit("No reconstructed cameras to check")
    print(
        f"{p.name}{'/' + tag if tag else ''}: Mapillary gravity against the reconstruction"
    )
    print(f"  {'leg':<8} {'sequence':<34} {'spread':>8} {'outliers':>9}")
    for r in rows:
        if r["spread_median_deg"] is None:
            print(
                f"  {r['leg']:<8} {r['folder'][:34]:<34}  no gravity (no Mapillary rotations)"
            )
        else:
            print(
                f"  {r['leg']:<8} {r['folder'][:34]:<34} {r['spread_median_deg']:7.2f}° "
                f"{r['outlier_cameras']:>4}/{r['cameras']}"
            )
    measured = [
        r["spread_median_deg"] for r in rows if r["spread_median_deg"] is not None
    ]
    if measured:
        print(
            f"  median over {len(measured)} legs and sequences: {np.median(measured):.2f}°"
        )
    out = p.results / "gravity_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"  wrote {out}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("--tag", help="Check the reconstruction of a tagged run")
    args = parser.parse_args()
    check(args.roi_dir, args.tag)


if __name__ == "__main__":
    main()
