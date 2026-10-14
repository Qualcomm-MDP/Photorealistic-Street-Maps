"""Stage 2 (Mapillary): fetch the images for a drawn route (ROI) into a new ROI folder.

For each line of the ROI (route_gui.py):
  1. Query Mapillary in small tiles along the line (a large bbox query returns only a sample
     of the images) and keep perspective (non-360) images inside the corridor, plus those
     up to 25 m beyond it, so a vehicle cutting a corner does not split its capture.
  2. Group by sequence (one capture session of one camera), drop near-duplicate frames taken
     while stopped (no baseline for SfM), and split each sequence into continuous runs at
     time gaps and position jumps (separate passes along the line), and trim each run to
     start and end inside the corridor. Turns are kept: the COLMAP layer splits the
     capture into straight legs itself.
  3. Take the best run: enough frames along its longest straight stretch first (what the
     COLMAP layer will reconstruct; frames in a turn do not count), then frames dense
     enough for COLMAP (median spacing <= --max-spacing), then the share of the drawn line
     it covers (in 25% steps), then frame spacing (in 1 m steps), then the most frames. --view keeps only cameras
     looking that way relative to travel; --sequence picks a sequence by id.
  4. Add every rig sibling: multi-camera rigs (e.g. a vehicle with left- and right-facing
     cameras) upload one sequence per camera; a sibling has the same make/model, overlaps
     the pick's capture window and was within 20 m of it at the same moments.
  5. Download each sequence to data/mapillary/<line>_cam<k>_<sequence id>/ as
     frame_0001.jpg, ... in capture order, with cameras.json holding each frame's Mapillary
     camera data (id, time, raw GPS, Mapillary's computed position and rotation, compass,
     altitude, calibration).

The ROI folder is the next free rois/mapillary_roi_N (N = 1, 2, ...); every later stage
writes into it too (data/ and results/). This stage writes data/mapillary/:
  roi.json        the drawn route
  sequences.json  one entry per downloaded sequence (folder, id, line, role, view, camera)

Needs a Mapillary client access token (https://www.mapillary.com/dashboard/developers):
  export MAPILLARY_TOKEN='MLY|...'

Usage:
  python mapillary_layer.py --roi routes/my_route.json
  python mapillary_layer.py --roi routes/my_route.json --view right --list
"""

import argparse
import json
import math
import os
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import (ROIS_DIR, USER_AGENT, LocalFrame, RoiPaths, get_json, line_length_m, next_roi_dir,
                    straight_stretches, wrap_deg)

MAPILLARY_URL = "https://graph.mapillary.com/images"
FIELDS = [
    "id", "sequence", "captured_at", "is_pano", "camera_type", "camera_parameters", "make", "model",
    "width", "height", "geometry", "computed_geometry", "compass_angle", "computed_compass_angle",
    "computed_rotation", "altitude", "computed_altitude", "thumb_2048_url", "thumb_original_url",
]
VIEWS = {"forward": 0.0, "right": 90.0, "back": 180.0, "left": -90.0}


# ----------------------------------------------------------------------------- search

def image_lonlat(img):
    """Mapillary's computed position (OpenSfM), or the raw GPS if there is none."""
    geom = img.get("computed_geometry") or img["geometry"]
    return geom["coordinates"]


def mapillary_images(token, bbox):
    params = urllib.parse.urlencode({"bbox": ",".join(f"{v:.6f}" for v in bbox),
                                     "fields": ",".join(FIELDS), "limit": 2000})
    data = get_json(f"{MAPILLARY_URL}?{params}", headers={"Authorization": f"OAuth {token}"})
    if len(data.get("data", [])) >= 2000:
        print(f"  warning: tile {bbox} hit the 2000-image limit; lower --tile")
    return data.get("data", [])


def tiles_along(segs, frame, tile, pad):
    """Lon/lat boxes of about tile x tile metres (plus pad) covering every line segment."""
    boxes = set()
    for a, b in segs:
        n = max(1, math.ceil(math.dist(a, b) / (tile / 2)))
        for k in range(n + 1):
            x = a[0] + (b[0] - a[0]) * k / n
            y = a[1] + (b[1] - a[1]) * k / n
            boxes.add((math.floor(x / tile), math.floor(y / tile)))
    out = []
    for i, j in sorted(boxes):
        x0, y0, x1, y1 = i * tile - pad, j * tile - pad, (i + 1) * tile + pad, (j + 1) * tile + pad
        out.append([*frame.lonlat(x0, y0), *frame.lonlat(x1, y1)])
    return out


def dist_to_segment(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / max(dx * dx + dy * dy, 1e-12)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def continuous_runs(imgs, frame, min_spacing, max_gap_s, max_jump_m):
    """Drop stationary duplicates, then split at capture-time gaps and position jumps."""
    runs, last = [], None
    for img in imgs:
        p = frame.xy(*image_lonlat(img))
        if last is not None:
            step = math.dist(p, last[1])
            if step < min_spacing:
                continue
            if (img["captured_at"] - last[0]["captured_at"]) / 1000 > max_gap_s or step > max_jump_m:
                last = None
        if last is None:
            runs.append([])
        runs[-1].append(img)
        last = (img, p)
    return runs


def view_angle(run, pts):
    """Median camera heading relative to the direction of travel, in degrees:
    0 = forward, +90 = right, -90 = left, +-180 = backward. None if unknown."""
    rel = []
    for k, img in enumerate(run):
        compass = img.get("computed_compass_angle", img.get("compass_angle"))
        a, b = pts[max(k - 1, 0)], pts[min(k + 1, len(pts) - 1)]
        if compass is None or math.dist(a, b) < 0.5:
            continue
        travel = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))  # clockwise from north
        rel.append(wrap_deg(compass - travel))
    if not rel:
        return None
    # Median of angles around the circular mean, fine for one camera's tight spread.
    ref = math.degrees(math.atan2(sum(math.sin(math.radians(r)) for r in rel),
                                  sum(math.cos(math.radians(r)) for r in rel)))
    return wrap_deg(ref + sorted(wrap_deg(r - ref) for r in rel)[len(rel) // 2])


def summarise(seq_id, run, frame, line_samples, reach):
    pts = [frame.xy(*image_lonlat(i)) for i in run]
    steps = sorted(math.dist(a, b) for a, b in zip(pts, pts[1:])) or [float("inf")]
    P = np.array(pts)
    covered = np.min(np.linalg.norm(line_samples[:, None] - P[None], axis=2), axis=1) <= reach
    first = run[0]
    return {
        "sequence": seq_id,
        "run": run,
        "images": len(run),
        # Frames COLMAP will get: the longest straight stretch (see common.straight_stretches).
        "straight": max((len(s) for s in straight_stretches(pts, [i["captured_at"] for i in run])), default=0),
        "spacing_m": steps[len(steps) // 2],
        "coverage": float(covered.mean()),
        "date": datetime.fromtimestamp(first["captured_at"] / 1000, timezone.utc).strftime("%Y-%m-%d"),
        "camera": f'{first.get("make", "?")} {first.get("model", "?")}',
        "view_deg": view_angle(run, pts),
    }


def find_runs(token, line, corridor, tile=80.0, min_spacing=1.0, max_gap_s=300.0, max_jump=40.0, slack=25.0):
    """Continuous runs of perspective images along one drawn line: (run summaries, frame).

    Runs are formed from images within corridor + slack of the line and then trimmed to
    start and end inside the corridor: where a vehicle cuts a corner, the frames just
    outside the corridor keep the run in one piece (without them the gap reads as a
    position jump and splits the capture at every corner)."""
    frame = LocalFrame(*line[0])
    xy = [frame.xy(*p) for p in line]
    segs = list(zip(xy, xy[1:]))
    images = {}
    tiles = tiles_along(segs, frame, tile, corridor + slack)
    for box in tiles:
        for img in mapillary_images(token, box):
            images[img["id"]] = img
    dist = {}
    for i in images.values():
        if not i.get("is_pano") and i.get("camera_type", "perspective") == "perspective":
            d = min(dist_to_segment(frame.xy(*image_lonlat(i)), a, b) for a, b in segs)
            if d <= corridor + slack:
                dist[i["id"]] = d
    print(f"  Mapillary: {len(images)} images from {len(tiles)} tiles, "
          f"{sum(d <= corridor for d in dist.values())} perspective images within {corridor:.0f} m of the line")

    # Points every 2 m along the line, to measure how much of it a run covers.
    samples = np.vstack([np.array(a) + np.linspace(0, 1, max(2, int(math.dist(a, b) / 2)))[:, None] * np.subtract(b, a)
                         for a, b in segs])
    by_seq = {}
    for img_id in dist:
        by_seq.setdefault(images[img_id]["sequence"], []).append(images[img_id])
    runs = []
    for seq_id, imgs in by_seq.items():
        imgs.sort(key=lambda i: i["captured_at"])
        for run in continuous_runs(imgs, frame, min_spacing, max_gap_s, max_jump):
            inside = [k for k, i in enumerate(run) if dist[i["id"]] <= corridor]
            if inside:
                runs.append(summarise(seq_id, run[inside[0]:inside[-1] + 1], frame, samples, corridor + 5))
    return runs, frame


def spacing_bucket(run):
    """Median spacing rounded down to whole metres; single-frame runs sort last."""
    return math.floor(run["spacing_m"]) if math.isfinite(run["spacing_m"]) else math.inf


def rank_runs(runs, min_images=20, max_spacing=6.0, view=None, view_tol=35.0):
    """Enough frames first, then frames dense enough for COLMAP (median spacing <=
    max_spacing; tested: a 7 m run split in COLMAP and fitted the walls 3.3 m off, a 5.4 m
    run of the same street 0.3 m), then coverage of the line in 25% steps, then spacing in
    1 m steps (2.1 vs 2.2 m is no real difference), then the most frames."""
    if view:
        runs = [r for r in runs if r["view_deg"] is not None
                and abs(wrap_deg(r["view_deg"] - VIEWS[view])) <= view_tol]
    return sorted(runs, key=lambda r: (r["straight"] < min_images, r["spacing_m"] > max_spacing,
                                       -min(int(r["coverage"] * 4), 3), spacing_bucket(r), -r["images"],
                                       r["spacing_m"]))


def print_runs(ranked, min_images, limit=10):
    print(f"\n  {'sequence':<24} {'images':>6} {'straight':>8} {'covers':>6} {'spacing m':>9} {'view':>6}  {'date':<10}  camera")
    for r in ranked[:limit]:
        flag = "" if r["straight"] >= min_images else "  (too few straight frames)"
        view = "?" if r["view_deg"] is None else f"{r['view_deg']:+.0f}"
        print(f"  {r['sequence']:<24} {r['images']:>6} {r['straight']:>8} {100 * r['coverage']:>5.0f}% {r['spacing_m']:>9.1f} {view:>6}  "
              f"{r['date']:<10}  {r['camera']}{flag}")


def rig_siblings(pick, runs, frame, max_dist=20.0, min_overlap=0.5, margin_s=1.0):
    """Runs of other sequences captured by the same rig at the same time as pick's frames:
    [(run summary, frames within pick's capture window)]."""
    t0, t1 = pick["run"][0]["captured_at"], pick["run"][-1]["captured_at"]
    pt = [i["captured_at"] for i in pick["run"]]
    pxy = np.array([frame.xy(*image_lonlat(i)) for i in pick["run"]])
    siblings = {}
    for r in runs:
        if r["sequence"] == pick["sequence"] or r["camera"] != pick["camera"]:
            continue
        inside = [i for i in r["run"] if t0 - margin_s * 1000 <= i["captured_at"] <= t1 + margin_s * 1000]
        if len(inside) < 5 or (inside[-1]["captured_at"] - inside[0]["captured_at"]) < min_overlap * (t1 - t0):
            continue
        dists = [math.dist((np.interp(i["captured_at"], pt, pxy[:, 0]), np.interp(i["captured_at"], pt, pxy[:, 1])),
                           frame.xy(*image_lonlat(i))) for i in inside]
        if np.median(dists) <= max_dist:
            best = siblings.get(r["sequence"])
            if best is None or len(inside) > len(best[1]):
                siblings[r["sequence"]] = (r, inside)
    return list(siblings.values())


# ----------------------------------------------------------------------------- download

def frame_record(img, name):
    """The camera data kept for one frame (cameras.json)."""
    lon, lat = image_lonlat(img)
    return {
        "file": name,
        "id": img["id"],
        "captured_at": datetime.fromtimestamp(img["captured_at"] / 1000, timezone.utc).isoformat(),
        "captured_at_ms": img["captured_at"],
        "lon": lon, "lat": lat,                             # Mapillary computed (OpenSfM), else GPS
        "gps": img["geometry"]["coordinates"],              # raw GPS [lon, lat]
        "computed": bool(img.get("computed_geometry")),
        "compass_angle": img.get("compass_angle"),          # raw compass (deg from north)
        "computed_compass_angle": img.get("computed_compass_angle"),
        "computed_rotation": img.get("computed_rotation"),  # OpenSfM world-to-camera, Rodrigues
        "altitude": img.get("altitude"),
        "computed_altitude": img.get("computed_altitude"),
        **{k: img.get(k) for k in ("camera_type", "camera_parameters", "make", "model", "width", "height")},
    }


def download_sequence(imgs, folder, header, resolution="2048", workers=8):
    """Download images as frame_0001.jpg, ... (capture order) plus cameras.json."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    url_field = "thumb_original_url" if resolution == "original" else "thumb_2048_url"
    names = [f"frame_{n:04d}.jpg" for n in range(1, len(imgs) + 1)]

    def fetch(job):
        img, name = job
        req = urllib.request.Request(img[url_field], headers={"User-Agent": USER_AGENT})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r, open(folder / name, "wb") as f:
                    shutil.copyfileobj(r, f)
                return
            except OSError:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, zip(imgs, names)))
    frames = [frame_record(img, name) for img, name in zip(imgs, names)]
    (folder / "cameras.json").write_text(json.dumps({**header, "frames": frames}, indent=1))
    print(f"  saved {len(imgs)} frames and cameras.json to {folder}")


def fetch_roi(roi, token, out=None, view=None, view_tol=35.0, sequence=None, min_images=20, max_spacing=6.0,
              max_images=None, no_rig=False, resolution="2048", list_only=False, rois_dir=ROIS_DIR):
    """Find and download the best run plus rig siblings for every line of the ROI.
    Returns the ROI folder (None with list_only)."""
    picks = []
    for k, line in enumerate(roi["lines"], start=1):
        print(f"Line {k}: {len(line)} points, {line_length_m(line):.0f} m, corridor {roi['corridor_m']:.0f} m")
        runs, frame = find_runs(token, line, roi["corridor_m"])
        ranked = rank_runs(runs, min_images, max_spacing, view, view_tol)
        if view:
            print(f"  {len(ranked)} runs with cameras looking {view} (within {view_tol:.0f} deg)")
        print_runs(ranked, min_images)
        if sequence:
            ranked = [r for r in ranked if r["sequence"] == sequence] or ranked
        if not ranked or ranked[0]["straight"] < min_images:
            print(f"  no run with {min_images}+ frames along a straight stretch on line {k}; skipping it "
                  "(lower --min-images, widen the corridor or draw another street)")
            continue
        picks.append((k, ranked[0], runs, frame))
    if list_only:
        return None
    if not picks:
        raise SystemExit("No line has a usable Mapillary run")

    out = Path(out) if out else next_roi_dir(rois_dir)
    paths = RoiPaths(out)
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.roi_json.write_text(json.dumps(roi, indent=1))
    sequences = []
    for k, pick, runs, frame in picks:
        chosen = pick["run"][:max_images] if max_images else pick["run"]
        pick = {**pick, "run": chosen}
        cams = [(pick, chosen)] + ([] if no_rig else rig_siblings(pick, runs, frame))
        for j, (r, imgs) in enumerate(cams):
            folder = f"line{k}_cam{j}_{r['sequence'][:8]}"
            role = "rig sibling" if j else "primary"
            view_deg = view_angle(imgs, [frame.xy(*image_lonlat(i)) for i in imgs])
            print(f"\n{folder}: sequence {r['sequence']} ({role}), "
                  f"view {'?' if view_deg is None else f'{view_deg:+.0f}'} deg, {len(imgs)} frames, {r['camera']}")
            header = {"roi": out.name, "line": k, "sequence": r["sequence"], "role": role, "camera": r["camera"],
                      "view_deg": view_deg, "date": r["date"], "resolution": resolution}
            download_sequence(imgs, paths.images / folder, header, resolution)
            sequences.append({"folder": folder, **{key: header[key] for key in ("sequence", "line", "role", "camera",
                                                                                 "view_deg", "date")},
                              "frames": len(imgs), "spacing_m": round(r["spacing_m"], 2),
                              "line_coverage": round(r["coverage"], 3) if j == 0 else None})
    paths.sequences_json.write_text(json.dumps(sequences, indent=1))
    print(f"\nROI folder {out}: {len(sequences)} sequence(s) on {len(picks)} line(s)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roi", type=Path, required=True, help="ROI file from route_gui.py (or a ROI folder)")
    parser.add_argument("--out", type=Path, help="ROI folder (default: next free rois/mapillary_roi_N)")
    parser.add_argument("--view", choices=list(VIEWS), help="Only start from cameras looking this way")
    parser.add_argument("--view-tol", type=float, default=35.0, help="Allowed deviation (deg) for --view")
    parser.add_argument("--sequence", help="Start from this Mapillary sequence instead of the best one")
    parser.add_argument("--min-images", type=int, default=20, help="Ignore runs with fewer frames")
    parser.add_argument("--max-spacing", type=float, default=6.0,
                        help="Prefer runs whose median frame spacing (m) is at most this")
    parser.add_argument("--max-images", type=int, help="Download at most this many frames per sequence")
    parser.add_argument("--no-rig", action="store_true", help="Do not add the rig's other cameras")
    parser.add_argument("--resolution", choices=["2048", "original"], default="2048")
    parser.add_argument("--list", action="store_true", help="Only list candidate runs")
    parser.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MAPILLARY_ACCESS_TOKEN"))
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("Set MAPILLARY_TOKEN (client access token from https://www.mapillary.com/dashboard/developers) "
                         "or pass --token")
    roi_file = args.roi / "roi.json" if args.roi.is_dir() else args.roi
    fetch_roi(json.loads(roi_file.read_text()), args.token, args.out, args.view, args.view_tol, args.sequence,
              args.min_images, args.max_spacing, args.max_images, args.no_rig, args.resolution, args.list)


if __name__ == "__main__":
    main()
