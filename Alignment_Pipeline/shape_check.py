"""Shape of each reconstructed leg, independent of the OSM fit (GPS is only the yardstick here).

Per leg and camera of a run:
  bow          lateral deviation of the camera path from a straight line (quadratic part, metres
               at the GPS scale), next to the same for Mapillary's raw GPS track: a straight
               street driven straight bows ~1 m; a bent reconstruction bows much more
  gravity      median angle between each camera's down direction and GeoCalib's (the model's
               consistency with an independent gravity estimate)
  scale        camera spacing in the last third of the leg / first third, both relative to GPS
               spacing (1.0 = no scale drift along the leg)
Writes nothing; prints one line per leg camera and a frame-weighted mean.

Usage: python shape_check.py ../Alignment_Pipeline/rois/mapillary_roi_11 [--tag mly60]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from align import level_with_gravity
from common import LocalFrame, RoiPaths, load_cameras, similarity_2d


def bow_of(P):
    c = P.mean(0)
    u = np.linalg.svd(P - c)[2][0]
    n = np.array([-u[1], u[0]])
    a, lat = (P - c) @ u, (P - c) @ n
    q = np.polyfit(a, lat, 2)
    return abs(q[0]) * ((a.max() - a.min()) / 2) ** 2, a, a.max() - a.min()


def leg_shape(p, cam):
    folder = cam["folder"]
    gc = json.loads(p.calibration_json(folder).read_text())
    meta = p.frames(folder)
    names, C, R = load_cameras(p.reconstruction / cam["model"])
    o, B, spread, _ = level_with_gravity(names, C, R, gc)
    L = (C - o) @ B.T
    enu = LocalFrame(*meta[names[0]]["gps"])
    g = np.array([enu.xy(*meta[n]["gps"]) for n in names])
    s = similarity_2d(L[:, :2], g)[0]
    bow, a, length = bow_of(L[:, :2] * s)
    gbow, ga, _ = bow_of(g)
    st, gs = np.abs(np.diff(a)), np.abs(np.diff(ga))
    k = max(len(st) // 3, 1)
    drift = (st[-k:].sum() / gs[-k:].sum()) / (st[:k].sum() / gs[:k].sum())
    return {"cameras": len(names), "length_m": length, "bow_m": bow, "gps_bow_m": gbow, "gravity_deg": spread,
            "scale_end_start": drift}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args()
    p = RoiPaths(args.roi_dir, args.tag)
    doc = json.loads(p.legs_json.read_text())
    rows = []
    for leg in doc["legs"]:
        for cam in leg["cameras"]:
            if not cam.get("model"):
                continue
            r = leg_shape(p, cam)
            rows.append(r)
            print(f"{leg['id']}/{cam['folder']}: {r['cameras']} cams over {r['length_m']:.0f} m, bow {r['bow_m']:.1f} m "
                  f"(GPS {r['gps_bow_m']:.1f} m), gravity {r['gravity_deg']:.1f} deg, scale end/start "
                  f"{r['scale_end_start']:.2f}, calibration {cam.get('calibration', {}).get('source')} "
                  f"{cam.get('calibration', {}).get('model')}{' refined' if cam.get('calibration', {}).get('refined') else ''}")
    w = np.array([r["cameras"] for r in rows], float)
    m = lambda k: float(np.average([r[k] for r in rows], weights=w))  # noqa: E731
    print(f"MEAN ({args.tag or 'base'}): bow {m('bow_m'):.1f} m (GPS {m('gps_bow_m'):.1f}), gravity {m('gravity_deg'):.1f} deg, "
          f"scale end/start {m('scale_end_start'):.2f}, {len(rows)} leg cameras")


if __name__ == "__main__":
    main()
