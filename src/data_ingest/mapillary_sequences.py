"""Pose alignment, stage 2 (Mapillary): the captures along an ROI's streets, downloaded
into a new ROI folder.

For each line of the ROI (route_gui.py, or pose_alignment.streets_from_bbox in the main
pipeline):
  1. Take Mapillary's images near the line: in the main pipeline the records the fetch stage
     got (ingest.py, which asks for ALIGNMENT_FIELDS when the poses are aligned); run on its
     own, a query through common.providers.MapillaryClient in small tiles along the line (a
     large bbox query returns only a sample of the images). Keep perspective and 360 images
     inside the corridor, plus those
     up to 25 m beyond it, so a vehicle cutting a corner does not split its capture.
  2. Group by sequence (one capture session of one camera), drop near-duplicate frames taken
     while stopped (no baseline for SfM), and split each sequence into continuous runs at
     time gaps and position jumps (separate passes along the line), and trim each run to
     start and end inside the corridor. Turns are kept: the reconstruct stage splits the
     capture into straight legs itself.
  3. Take the best run: enough frames along its longest straight stretch first (frames in a
     turn do not count), then frames dense enough to reconstruct (median spacing <=
     --max-spacing), then the share of the drawn line it covers (in 25% steps), then which way
     the camera looks: a 360 run (cut into right and left views) first, then a camera facing
     the side of the street, then one facing forward or back (it sees the facades at a grazing
     angle), then frame spacing (in 1 m steps), then the most frames (--pano prefer: a 360
     run whenever it is long and dense enough, before coverage; --pano off: never). --view keeps only cameras looking that way relative to travel (a 360
     run counts for left and right); --sequence picks a sequence by id.
  4. Add every rig sibling: multi-camera rigs (e.g. a vehicle with left- and right-facing
     cameras) upload one sequence per camera; a sibling has the same make/model, overlaps
     the pick's capture window and was within 20 m of it at the same moments.
     A 360 pick has no siblings: it becomes its own two-camera rig instead (see 5).
  5. Download each sequence to data/mapillary/<line>_cam<k>_<sequence id>/ as
     frame_0001.jpg, ... in capture order, with cameras.json holding each frame's Mapillary
     camera data (id, time, raw GPS, Mapillary's computed position and rotation, compass,
     altitude, calibration).
     A 360 sequence is downloaded at full resolution and each equirectangular frame is cut
     into two pinhole views looking right and left of travel (--pano-fov degrees square,
     default 90; at most 2048 px): <line>_cam0_<id> (right, the primary) and <line>_cam1_<id>
     (left, a rig sibling). The crop directions are fixed in the 360 image (the camera is
     rigid on the vehicle): +-90 deg from travel by the run's median compass. Each crop's
     cameras.json is that of a perspective camera with an exact, distortion-free focal
     (camera_parameters [focal, 0, 0]), its compass and rotation turned to the crop's
     direction, and a "pano" header (side, yaw in the 360 image, field of view, source
     size); later stages use that focal as is (calibrate.py, reconstruct.py).
The ROI folder is the next free outputs/alignment/mapillary_roi_N (N = 1, 2, ...); every later stage
writes into it too (data/ and results/). This stage writes data/mapillary/:
  roi.json        the drawn route
  sequences.json  one entry per downloaded sequence (folder, id, line, role, view, camera)

With keep_originals (the main pipeline), each photo is downloaded once at full resolution
into data/mapillary/originals/ (data_ingest.image_store), the frames are scaled from it to
2048 px, and texturing reuses the stored photos.

Needs a Mapillary client access token (https://www.mapillary.com/dashboard/developers):
  export MAPILLARY_ACCESS_TOKEN='MLY|...'     (or MAPILLARY_TOKEN)

Usage:
  python -m data_ingest.mapillary_sequences --roi outputs/alignment/routes/my_route.json
  python -m data_ingest.mapillary_sequences --roi outputs/alignment/routes/my_route.json --view right --list
  python -m data_ingest.mapillary_sequences --roi outputs/alignment/mapillary_roi_18 --out outputs/alignment/mapillary_roi_19   # reuse an ROI's route
"""

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from common.constants import BoundingBox
from common.providers.mapillary import MapillaryClient
from data_ingest.image_store import ImageStore, fetch_bytes
from data_ingest.pano import (
    PANO_SIDES,
    crop_rotvec,
    crop_size,
    crop_view,
    is_pano,
)
from pose_alignment.geo_common import (
    ROIS_DIR,
    LocalFrame,
    RoiPaths,
    line_length_m,
    next_roi_dir,
    straight_stretches,
    wrap_deg,
)

# The Mapillary fields this stage needs (ingest.py adds them to its own when aligning).
ALIGNMENT_FIELDS = FIELDS = [
    "id",
    "sequence",
    "captured_at",
    "is_pano",
    "camera_type",
    "camera_parameters",
    "make",
    "model",
    "width",
    "height",
    "geometry",
    "computed_geometry",
    "compass_angle",
    "computed_compass_angle",
    "computed_rotation",
    "altitude",
    "computed_altitude",
    "thumb_2048_url",
    "thumb_original_url",
]
VIEWS = {"forward": 0.0, "right": 90.0, "back": 180.0, "left": -90.0}


# ----------------------------------------------------------------------------- search


def image_lonlat(img):
    """Mapillary's computed position (OpenSfM), or the raw GPS if there is none."""
    geom = img.get("computed_geometry") or img["geometry"]
    return geom["coordinates"]


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
        x0, y0, x1, y1 = (
            i * tile - pad,
            j * tile - pad,
            (i + 1) * tile + pad,
            (j + 1) * tile + pad,
        )
        out.append([*frame.lonlat(x0, y0), *frame.lonlat(x1, y1)])
    return out


def dist_to_segment(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(
        0.0,
        min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / max(dx * dx + dy * dy, 1e-12)),
    )
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
            if (
                img["captured_at"] - last[0]["captured_at"]
            ) / 1000 > max_gap_s or step > max_jump_m:
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
        travel = math.degrees(
            math.atan2(b[0] - a[0], b[1] - a[1])
        )  # clockwise from north
        rel.append(wrap_deg(compass - travel))
    if not rel:
        return None
    # Median of angles around the circular mean, fine for one camera's tight spread.
    ref = math.degrees(
        math.atan2(
            sum(math.sin(math.radians(r)) for r in rel),
            sum(math.cos(math.radians(r)) for r in rel),
        )
    )
    return wrap_deg(ref + sorted(wrap_deg(r - ref) for r in rel)[len(rel) // 2])


def summarise(seq_id, run, frame, line_samples, reach):
    pts = [frame.xy(*image_lonlat(i)) for i in run]
    steps = sorted(math.dist(a, b) for a, b in zip(pts, pts[1:])) or [float("inf")]
    P = np.array(pts)
    covered = (
        np.min(np.linalg.norm(line_samples[:, None] - P[None], axis=2), axis=1) <= reach
    )
    first = run[0]
    return {
        "sequence": seq_id,
        "run": run,
        "images": len(run),
        # Frames COLMAP will get: the longest straight stretch (see common.straight_stretches).
        "straight": max(
            (len(s) for s in straight_stretches(pts, [i["captured_at"] for i in run])),
            default=0,
        ),
        "spacing_m": steps[len(steps) // 2],
        "coverage": float(covered.mean()),
        "date": datetime.fromtimestamp(
            first["captured_at"] / 1000, timezone.utc
        ).strftime("%Y-%m-%d"),
        "camera": f"{first.get('make', '?')} {first.get('model', '?')}",
        "pano": is_pano(first),
        "view_deg": view_angle(run, pts),  # 360: where the image centre looks
    }


def line_images(line, corridor, tile=80.0, slack=25.0, client=None):
    """{id: Mapillary record} of the images in tiles of about tile x tile metres along the line
    (plus corridor + slack), from Mapillary's API."""
    client = client or MapillaryClient()
    frame = LocalFrame(*line[0])
    xy = [frame.xy(*p) for p in line]
    images = {}
    tiles = tiles_along(list(zip(xy, xy[1:])), frame, tile, corridor + slack)
    for w, s, e, n in tiles:
        # one request per tile (split only when Mapillary's 2000-image cap is near)
        box = BoundingBox(min_lat=s, max_lat=n, min_lon=w, max_lon=e)
        for img in client.fetch_images(box, FIELDS, tile_deg=1.0):
            images[img["id"]] = img
    print(f"  Mapillary: {len(images)} images from {len(tiles)} tiles")
    return images


def find_runs(
    images,
    line,
    corridor,
    tile=80.0,
    min_spacing=1.0,
    max_gap_s=300.0,
    max_jump=40.0,
    slack=25.0,
):
    """Continuous runs of perspective or 360 images along one drawn line: (run summaries,
    frame, points every 2 m along the line). images: {id: Mapillary record} near the line
    (line_images, or the fetch stage's records).

    Runs are formed from images within corridor + slack of the line and then trimmed to
    start and end inside the corridor: where a vehicle cuts a corner, the frames just
    outside the corridor keep the run in one piece (without them the gap reads as a
    position jump and splits the capture at every corner)."""
    frame = LocalFrame(*line[0])
    xy = [frame.xy(*p) for p in line]
    segs = list(zip(xy, xy[1:]))
    dist = {}
    for i in images.values():
        if is_pano(i) or i.get("camera_type", "perspective") == "perspective":
            d = min(dist_to_segment(frame.xy(*image_lonlat(i)), a, b) for a, b in segs)
            if d <= corridor + slack:
                dist[i["id"]] = d
    print(
        f"  Mapillary: "
        f"{sum(d <= corridor and not is_pano(images[k]) for k, d in dist.items())} perspective and "
        f"{sum(d <= corridor and is_pano(images[k]) for k, d in dist.items())} 360 images within {corridor:.0f} m of the line"
    )

    # Points every 2 m along the line, to measure how much of it a run covers.
    samples = np.vstack(
        [
            np.array(a)
            + np.linspace(0, 1, max(2, int(math.dist(a, b) / 2)))[:, None]
            * np.subtract(b, a)
            for a, b in segs
        ]
    )
    by_seq = {}
    for img_id in dist:
        by_seq.setdefault(images[img_id]["sequence"], []).append(images[img_id])
    runs = []
    for seq_id, imgs in by_seq.items():
        imgs.sort(key=lambda i: i["captured_at"])
        for run in continuous_runs(imgs, frame, min_spacing, max_gap_s, max_jump):
            inside = [k for k, i in enumerate(run) if dist[i["id"]] <= corridor]
            if inside:
                runs.append(
                    summarise(
                        seq_id,
                        run[inside[0] : inside[-1] + 1],
                        frame,
                        samples,
                        corridor + 5,
                    )
                )
    return runs, frame, samples


def spacing_bucket(run):
    """Median spacing rounded down to whole metres; single-frame runs sort last."""
    return math.floor(run["spacing_m"]) if math.isfinite(run["spacing_m"]) else math.inf


def view_rank(run, side_deg=45.0):
    """0 a 360 run, 1 a camera facing the side of the street (45-135 deg from travel), 2 one
    facing forward or back, or of unknown direction."""
    if run["pano"]:
        return 0
    v = run.get("view_deg")
    return 1 if v is not None and side_deg <= abs(wrap_deg(v)) <= 180 - side_deg else 2


def rank_runs(
    runs, min_images=20, max_spacing=10.0, view=None, view_tol=35.0, pano="auto"
):
    """Enough frames first, then frames dense enough for COLMAP (median spacing <=
    max_spacing: 10 m, as the reconstruct stage matches widely spaced frames with LightGlue: roi_4
    at 8.6 m and roi_5 at 9.9 m came out reliable; roi_9 at 10.3 m, a forward camera with
    buildings far back, did not), then coverage of the line in 25% steps, then spacing in
    1 m steps (2.1 vs 2.2 m is no real difference), then the most frames. Among runs covering
    the line equally, the camera's direction decides first (view_rank): a 360 run (right and
    left views, both facing the facades) before a perspective camera facing the side of the
    street, before one facing forward or back, which sees the facades at a grazing angle
    (roi_3 of 2026-11-04: forward cameras beat 360 runs of the same street and 3 of their 8
    legs came out unreliable, every 360 view reliable).
    pano: "auto" ranks 360 runs as above, "prefer" takes one whenever it has enough dense
    frames (before coverage), "off" never."""
    if pano == "off":
        runs = [r for r in runs if not r["pano"]]
    if view:
        # a 360 run gives right and left views, never forward or back
        runs = [
            r
            for r in runs
            if (
                view in PANO_SIDES
                if r["pano"]
                else r["view_deg"] is not None
                and abs(wrap_deg(r["view_deg"] - VIEWS[view])) <= view_tol
            )
        ]
    return sorted(
        runs,
        key=lambda r: (
            r["straight"] < min_images,
            r["spacing_m"] > max_spacing,
            pano == "prefer" and not r["pano"],
            -min(int(r["coverage"] * 4), 3),
            view_rank(r),
            spacing_bucket(r),
            -r["images"],
            r["spacing_m"],
        ),
    )


def covered(run, frame, samples, reach):
    """Which of the line's sample points the run's frames are within reach of."""
    P = np.array([frame.xy(*image_lonlat(i)) for i in run["run"]])
    return np.min(np.linalg.norm(samples[:, None] - P[None], axis=2), axis=1) <= reach


def choose_runs(
    ranked,
    frame,
    samples,
    reach,
    min_images=20,
    max_spacing=10.0,
    min_new=0.15,
    min_new_m=30.0,
):
    """Runs that together cover the line: the best ranked run first, then, while some of the
    line is still uncovered, the run that covers most of the rest (in 25% steps of the line;
    the ranking breaks ties). A further run needs min_images frames along a straight stretch,
    a median spacing of at most max_spacing, and to cover at least min_new of the line and
    min_new_m metres not covered yet, so where one capture ends another takes over."""
    if not ranked or ranked[0]["straight"] < min_images:
        return []
    picks = [ranked[0]]
    done = covered(ranked[0], frame, samples, reach)
    step = (
        float(np.median(np.linalg.norm(np.diff(samples, axis=0), axis=1)))
        if len(samples) > 1
        else 2.0
    )
    while True:
        best = None
        for k, r in enumerate(ranked):
            if any(r is q for q in picks) or r["straight"] < min_images:
                continue
            if r["spacing_m"] > max_spacing:
                continue
            new = covered(r, frame, samples, reach) & ~done
            share = float(new.mean())
            if share < min_new or new.sum() * step < min_new_m:
                continue
            key = (-min(int(share * 4), 3), k)
            if best is None or key < best[0]:
                best = (key, r, new)
        if best is None:
            return picks
        picks.append(best[1])
        done |= best[2]


def print_runs(ranked, min_images, limit=10):
    print(
        f"\n  {'sequence':<24} {'images':>6} {'straight':>8} {'covers':>6} {'spacing m':>9} {'view':>6}  {'date':<10}  camera"
    )
    for r in ranked[:limit]:
        flag = "" if r["straight"] >= min_images else "  (too few straight frames)"
        view = "?" if r["view_deg"] is None else f"{r['view_deg']:+.0f}"
        print(
            f"  {r['sequence']:<24} {r['images']:>6} {r['straight']:>8} {100 * r['coverage']:>5.0f}% {r['spacing_m']:>9.1f} {view:>6}  "
            f"{r['date']:<10}  {r['camera']}{' (360)' if r['pano'] else ''}{flag}"
        )


def rig_siblings(pick, runs, frame, max_dist=20.0, min_overlap=0.5, margin_s=1.0):
    """Runs of other sequences captured by the same rig at the same time as pick's frames:
    [(run summary, frames within pick's capture window)]."""
    t0, t1 = pick["run"][0]["captured_at"], pick["run"][-1]["captured_at"]
    pt = [i["captured_at"] for i in pick["run"]]
    pxy = np.array([frame.xy(*image_lonlat(i)) for i in pick["run"]])
    siblings = {}
    for r in runs:
        if (
            r["sequence"] == pick["sequence"]
            or r["camera"] != pick["camera"]
            or r["pano"]
        ):
            continue
        inside = [
            i
            for i in r["run"]
            if t0 - margin_s * 1000 <= i["captured_at"] <= t1 + margin_s * 1000
        ]
        if len(inside) < 5 or (
            inside[-1]["captured_at"] - inside[0]["captured_at"]
        ) < min_overlap * (t1 - t0):
            continue
        dists = [
            math.dist(
                (
                    np.interp(i["captured_at"], pt, pxy[:, 0]),
                    np.interp(i["captured_at"], pt, pxy[:, 1]),
                ),
                frame.xy(*image_lonlat(i)),
            )
            for i in inside
        ]
        if np.median(dists) <= max_dist:
            best = siblings.get(r["sequence"])
            if best is None or len(inside) > len(best[1]):
                siblings[r["sequence"]] = (r, inside)
    return list(siblings.values())


# ----------------------------------------------------------------------------- download


def frame_record(img, name, crop=None):
    """The camera data kept for one frame (cameras.json). crop: a 360 frame's pinhole view
    ({yaw_deg, focal_norm, width, height}), whose camera data replaces the 360 camera's."""
    lon, lat = image_lonlat(img)
    rec = {
        "file": name,
        "id": img["id"],
        "captured_at": datetime.fromtimestamp(
            img["captured_at"] / 1000, timezone.utc
        ).isoformat(),
        "captured_at_ms": img["captured_at"],
        "lon": lon,
        "lat": lat,  # Mapillary computed (OpenSfM), else GPS
        "gps": img["geometry"]["coordinates"],  # raw GPS [lon, lat]
        "computed": bool(img.get("computed_geometry")),
        "compass_angle": img.get("compass_angle"),  # raw compass (deg from north)
        "computed_compass_angle": img.get("computed_compass_angle"),
        "computed_rotation": img.get(
            "computed_rotation"
        ),  # OpenSfM world-to-camera, Rodrigues
        "altitude": img.get("altitude"),
        "computed_altitude": img.get("computed_altitude"),
        **{
            k: img.get(k)
            for k in (
                "camera_type",
                "camera_parameters",
                "make",
                "model",
                "width",
                "height",
            )
        },
    }
    if crop:
        yaw = crop["yaw_deg"]
        turn = lambda a: None if a is None else (a + yaw) % 360  # noqa: E731
        rot = img.get("computed_rotation")
        if rot:
            rot = crop_rotvec(rot, yaw)
        rec.update(
            camera_type="perspective",
            camera_parameters=[crop["focal_norm"], 0.0, 0.0],
            width=crop["width"],
            height=crop["height"],
            compass_angle=turn(img.get("compass_angle")),
            computed_compass_angle=turn(img.get("computed_compass_angle")),
            computed_rotation=rot,
            pano_yaw_deg=round(yaw, 2),
        )
    return rec


def skipped_note(n_all, n_saved):
    return (
        f" ({n_all - n_saved} that Mapillary could not serve skipped)"
        if n_saved < n_all
        else ""
    )


def frame_bytes(data, max_side):
    """A downloaded photo as a frame: as is, or scaled down to max_side px (long side)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or max(img.shape[:2]) <= max_side:
        return data
    k = max_side / max(img.shape[:2])
    small = cv2.resize(
        img,
        (round(img.shape[1] * k), round(img.shape[0] * k)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def download_sequence(
    imgs, folder, header, resolution="2048", workers=8, store: ImageStore | None = None
):
    """Download images as frame_0001.jpg, ... (capture order) plus cameras.json. Images
    Mapillary cannot serve are skipped; returns the images saved. With a store, each photo
    is downloaded at full resolution into it (or taken from it) and the frame is scaled
    from that to 2048 px (resolution "2048", the size of Mapillary's thumb_2048_url)."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    url_field = "thumb_original_url" if resolution == "original" else "thumb_2048_url"

    def fetch(img):
        if store is None:
            return fetch_bytes(img[url_field], 120)
        data = store.get(
            str(img["id"]), img.get("thumb_original_url") or img[url_field]
        )
        return (
            data
            if data is None or resolution == "original"
            else frame_bytes(data, 2048)
        )

    with ThreadPoolExecutor(workers) as pool:
        data = list(pool.map(fetch, imgs))
    saved = [(img, d) for img, d in zip(imgs, data) if d is not None]
    names = [f"frame_{n:04d}.jpg" for n in range(1, len(saved) + 1)]
    for name, (_, d) in zip(names, saved):
        (folder / name).write_bytes(d)
    frames = [frame_record(img, name) for name, (img, _) in zip(names, saved)]
    (folder / "cameras.json").write_text(
        json.dumps({**header, "frames": frames}, indent=1)
    )
    print(
        f"  saved {len(saved)} frames and cameras.json to {folder}{skipped_note(len(imgs), len(saved))}"
    )
    return [img for img, _ in saved]


def download_pano(
    imgs,
    folders,
    headers,
    view_deg,
    fov_deg=90.0,
    workers=6,
    store: ImageStore | None = None,
):
    """Download 360 frames at full resolution and save each as pinhole crops looking right and left
    of travel (PANO_SIDES), one folder per side (frame_0001.jpg, ... plus cameras.json). The
    crop directions are fixed in the 360 image: the side's angle minus the run's median view_deg
    (where the image centre looks relative to travel; 0 when there is no compass). Returns
    ({side: yaw in the 360 image}, the images saved: ones Mapillary cannot serve are skipped)."""
    yaws = {side: wrap_deg(a - (view_deg or 0.0)) for side, a in PANO_SIDES.items()}
    for folder in folders.values():
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    sizes = {}
    # One crop size for the whole run (COLMAP takes each folder as one camera), from the 360 images' width
    widths = [i["width"] for i in imgs if i.get("width")]
    size = crop_size(np.median(widths) if widths else None, fov_deg)

    def fetch(img):
        url = img.get("thumb_original_url") or img["thumb_2048_url"]
        return store.get(str(img["id"]), url, 180) if store else fetch_bytes(url, 180)

    def crop(job):
        name, data = job
        pano = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        sizes[name] = pano.shape[1::-1]
        crops = {}
        for side, yaw in yaws.items():
            cv2.imwrite(
                str(folders[side] / name),
                crop_view(pano, yaw, fov_deg, size),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            crops[side] = {
                "yaw_deg": yaw,
                "width": size,
                "height": size,
                "focal_norm": 0.5 / math.tan(math.radians(fov_deg) / 2),
            }
        return crops

    with ThreadPoolExecutor(workers) as pool:
        data = list(pool.map(fetch, imgs))
    n_all, saved = len(imgs), [(img, d) for img, d in zip(imgs, data) if d is not None]
    imgs = [img for img, _ in saved]
    names = [f"frame_{n:04d}.jpg" for n in range(1, len(saved) + 1)]
    with ThreadPoolExecutor(workers) as pool:
        crops = list(pool.map(crop, zip(names, [d for _, d in saved])))
    W, H = max(sizes.values())
    if size < 1024:
        print(f"  warning: 360 frames only {W} px wide here; the crops are {size} px")
    for side, folder in folders.items():
        frames = [
            frame_record(img, name, c[side]) for img, name, c in zip(imgs, names, crops)
        ]
        pano = {
            "side": side,
            "yaw_deg": round(yaws[side], 2),
            "fov_deg": fov_deg,
            "source_width": W,
            "source_height": H,
            "source_camera_type": imgs[0].get("camera_type") or "spherical",
        }
        (folder / "cameras.json").write_text(
            json.dumps({**headers[side], "pano": pano, "frames": frames}, indent=1)
        )
        print(
            f"  saved {len(imgs)} {side} crops ({crops[0][side]['width']} px, {fov_deg:g} deg, "
            f"{yaws[side]:+.0f} deg from the 360 image's centre) and cameras.json to {folder}"
            f"{skipped_note(n_all, len(imgs))}"
        )
    return yaws, imgs


TIMINGS = {
    "search_s": 0.0,
    "download_s": 0.0,
    "frames": 0,
}  # filled by fetch_roi, read by run_pipeline


def choose_roi(
    roi,
    images=None,
    view=None,
    view_tol=35.0,
    sequence=None,
    min_images=20,
    max_spacing=10.0,
    no_rig=False,
    pano="auto",
    one_run=False,
):
    """The captures to use for every line of the ROI: a JSON-able selection {"roi", "groups"}.
    Each group is one chosen run ("pick": its summary, with "run" its records) plus its rig
    siblings ("siblings": [(summary, records in the pick's capture window)]); "line" numbers
    the groups, "street_line" is the ROI line it covers. A line gets several groups when one
    capture does not cover it (choose_runs; one_run: only the best run, as before).

    images: Mapillary records (ALIGNMENT_FIELDS) to choose from, e.g. the fetch stage's;
    without them each line's images are queried (MAPILLARY_ACCESS_TOKEN)."""
    if images is not None and not isinstance(images, dict):
        images = {img["id"]: img for img in images}
    groups = []
    seen: dict = {}  # every record looked at, to fill the gap when merging runs
    for k, line in enumerate(roi["lines"], start=1):
        print(
            f"Line {k}: {len(line)} points, {line_length_m(line):.0f} m, corridor {roi['corridor_m']:.0f} m"
        )
        t0 = time.time()
        near = images if images is not None else line_images(line, roi["corridor_m"])
        seen.update(near)
        runs, frame, samples = find_runs(near, line, roi["corridor_m"])
        TIMINGS["search_s"] += time.time() - t0
        ranked = rank_runs(runs, min_images, max_spacing, view, view_tol, pano)
        if view:
            print(
                f"  {len(ranked)} runs with cameras looking {view} (within {view_tol:.0f} deg)"
            )
        print_runs(ranked, min_images)
        if sequence:
            ranked = [r for r in ranked if r["sequence"] == sequence] or ranked
        picks = choose_runs(
            ranked, frame, samples, roi["corridor_m"] + 5, min_images, max_spacing
        )
        if one_run:
            picks = picks[:1]
        if not picks:
            print(
                f"  no run with {min_images}+ frames along a straight stretch on line {k}; skipping it "
                "(lower --min-images, widen the corridor or draw another street)"
            )
            continue
        if len(picks) > 1:
            print(
                f"  {len(picks)} runs together cover {100 * float(np.logical_or.reduce([covered(r, frame, samples, roi['corridor_m'] + 5) for r in picks]).mean()):.0f}% "
                f"of the line: " + ", ".join(r["sequence"] for r in picks)
            )
        taken = {r["sequence"] for r in picks}
        for pick in picks:
            siblings = (
                []
                if no_rig or pick["pano"]
                else [
                    (r, imgs)
                    for r, imgs in rig_siblings(pick, runs, frame)
                    if r["sequence"] not in taken
                ]
            )
            groups.append(
                {
                    "line": len(groups) + 1,
                    "street_line": k,
                    "pick": pick,
                    "siblings": siblings,
                }
            )
    return {"roi": roi, "groups": merge_repeated(groups, seen)}


def _span(imgs):
    return imgs[0]["captured_at"], imgs[-1]["captured_at"]


def _joined(a, b, records, min_spacing=1.0):
    """The frames of runs a and b of one sequence plus that sequence's frames between them
    (records), in capture order, leaving out frames under min_spacing metres from the last
    one kept (taken while stopped: no baseline for SfM)."""
    t0 = min(_span(a)[0], _span(b)[0])
    t1 = max(_span(a)[1], _span(b)[1])
    seq = a[0]["sequence"]
    pool = {i["id"]: i for i in [*a, *b]}
    for img in records.values():
        if img.get("sequence") == seq and t0 <= img.get("captured_at", -1) <= t1:
            pool.setdefault(img["id"], img)
    ordered = sorted(pool.values(), key=lambda i: i["captured_at"])
    frame = LocalFrame(*image_lonlat(ordered[0]))
    kept, last = [], None
    for img in ordered:
        xy = frame.xy(*image_lonlat(img))
        if last is None or math.dist(xy, last) >= min_spacing:
            kept.append(img)
            last = xy
    return kept


def merge_repeated(groups, records, max_gap_s=60.0):
    """Groups whose picks are the same capture (one sequence, its runs overlapping in time or
    less than max_gap_s apart) chosen on several streets, merged into one: one run through
    the turn between the streets, reconstructed (and calibrated) once, its legs linked
    across the turn like any capture's. The frames between the runs come from records;
    rig siblings are merged the same way. "street_line" stays the first street;
    "street_lines" lists them all."""
    merged: list[dict] = []
    for g in groups:
        pick = g["pick"]
        a0, a1 = _span(pick["run"])
        into = None
        for h in merged:
            hp = h["pick"]
            if hp["sequence"] != pick["sequence"] or hp["pano"] != pick["pano"]:
                continue
            b0, b1 = _span(hp["run"])
            if max(a0, b0) - min(a1, b1) <= max_gap_s * 1000:
                into = h
                break
        if into is None:
            merged.append({**g, "street_lines": [g["street_line"]]})
            continue
        run = _joined(into["pick"]["run"], pick["run"], records)
        frame = LocalFrame(*image_lonlat(run[0]))
        pts = [frame.xy(*image_lonlat(i)) for i in run]
        steps = sorted(math.dist(p, q) for p, q in zip(pts, pts[1:])) or [float("inf")]
        into["pick"] = {
            **into["pick"],
            "run": run,
            "images": len(run),
            "spacing_m": steps[len(steps) // 2],
            "coverage": max(into["pick"]["coverage"], pick["coverage"]),
        }
        sibs = {r["sequence"]: (r, imgs) for r, imgs in into["siblings"]}
        for r, imgs in g["siblings"]:
            if r["sequence"] in sibs:
                r0, imgs0 = sibs[r["sequence"]]
                sibs[r["sequence"]] = (r0, _joined(imgs0, imgs, records))
            else:
                sibs[r["sequence"]] = (r, imgs)
        into["siblings"] = list(sibs.values())
        into["street_lines"].append(g["street_line"])
        print(
            f"  {pick['sequence']} was chosen on lines {', '.join(map(str, into['street_lines']))}: "
            f"merged into one run of {len(run)} frames"
        )
    for n, g in enumerate(merged, start=1):
        g["line"] = n
    return merged


def selection_images(selection):
    """Every Mapillary record a selection downloads (picks and rig siblings)."""
    out = {}
    for g in selection["groups"]:
        for img in g["pick"]["run"]:
            out[img["id"]] = img
        for _, imgs in g["siblings"]:
            for img in imgs:
                out[img["id"]] = img
    return list(out.values())


def download_selection(
    selection,
    out=None,
    max_images=None,
    resolution="2048",
    rois_dir=ROIS_DIR,
    pano_fov=90.0,
    keep_originals=False,
):
    """Download a selection (choose_roi) into a new ROI folder; returns the folder.
    keep_originals: download full-resolution photos into data/mapillary/originals/ (see the
    module docstring)."""
    if not selection["groups"]:
        raise SystemExit("No line has a usable Mapillary run")
    out = Path(out) if out else next_roi_dir(rois_dir)
    paths = RoiPaths(out)
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.roi_json.write_text(json.dumps(selection["roi"], indent=1))
    store = ImageStore(paths.originals) if keep_originals else None
    sequences = []
    for g in selection["groups"]:
        n, pick = g["line"], g["pick"]
        chosen = pick["run"][:max_images] if max_images else pick["run"]
        pick = {**pick, "run": chosen}
        if pick["pano"]:
            sequences += fetch_pano(paths, out, n, pick, pano_fov, store)
            continue
        for j, (r, imgs) in enumerate([(pick, chosen)] + list(g["siblings"])):
            folder = f"line{n}_cam{j}_{r['sequence'][:8]}"
            role = "rig sibling" if j else "primary"
            frame = LocalFrame(*image_lonlat(imgs[0]))
            view_deg = view_angle(imgs, [frame.xy(*image_lonlat(i)) for i in imgs])
            print(
                f"\n{folder}: sequence {r['sequence']} ({role}), "
                f"view {'?' if view_deg is None else f'{view_deg:+.0f}'} deg, {len(imgs)} frames, {r['camera']}"
            )
            header = {
                "roi": out.name,
                "line": n,
                "street_line": g["street_line"],
                "sequence": r["sequence"],
                "role": role,
                "camera": r["camera"],
                "view_deg": view_deg,
                "date": r["date"],
                "resolution": resolution,
            }
            t0 = time.time()
            imgs = download_sequence(
                imgs, paths.images / folder, header, resolution, store=store
            )
            TIMINGS["download_s"] += time.time() - t0
            TIMINGS["frames"] += len(imgs)
            sequences.append(
                {
                    "folder": folder,
                    **{
                        key: header[key]
                        for key in (
                            "sequence",
                            "line",
                            "street_line",
                            "role",
                            "camera",
                            "view_deg",
                            "date",
                        )
                    },
                    "frames": len(imgs),
                    "spacing_m": round(r["spacing_m"], 2),
                    "line_coverage": round(r["coverage"], 3)
                    if role != "rig sibling"
                    else None,
                }
            )
    paths.sequences_json.write_text(json.dumps(sequences, indent=1))
    print(
        f"\nROI folder {out}: {len(sequences)} sequence(s) in {len(selection['groups'])} "
        f"run(s) on {len({g['street_line'] for g in selection['groups']})} line(s)"
    )
    return out


def fetch_roi(
    roi,
    token=None,
    out=None,
    view=None,
    view_tol=35.0,
    sequence=None,
    min_images=20,
    max_spacing=10.0,
    max_images=None,
    no_rig=False,
    resolution="2048",
    list_only=False,
    rois_dir=ROIS_DIR,
    pano="auto",
    pano_fov=90.0,
    images=None,
    keep_originals=False,
    one_run=False,
):
    """Choose (choose_roi) and download (download_selection) the captures of every line of
    the ROI. Returns the ROI folder (None with list_only)."""
    if token and not os.environ.get("MAPILLARY_ACCESS_TOKEN"):
        os.environ["MAPILLARY_ACCESS_TOKEN"] = token  # MapillaryClient reads it there
    selection = choose_roi(
        roi,
        images,
        view,
        view_tol,
        sequence,
        min_images,
        max_spacing,
        no_rig,
        pano,
        one_run,
    )
    if list_only:
        return None
    return download_selection(
        selection, out, max_images, resolution, rois_dir, pano_fov, keep_originals
    )


def fetch_pano(paths, out, n, pick, fov_deg, store=None):
    """Download a 360 run as two crop folders (right = primary, left = rig sibling); their
    sequences.json entries."""
    imgs = pick["run"]
    frame = LocalFrame(*image_lonlat(imgs[0]))
    view_deg = view_angle(imgs, [frame.xy(*image_lonlat(i)) for i in imgs])
    folders = {
        side: paths.images / f"line{n}_cam{j}_{pick['sequence'][:8]}"
        for j, side in enumerate(PANO_SIDES)
    }
    print(
        f"\nline {n}: 360 sequence {pick['sequence']}, {len(imgs)} frames, {pick['camera']}; image centre looks "
        f"{'?' if view_deg is None else f'{view_deg:+.0f}'} deg from travel; cutting right and left views"
    )
    headers = {}
    for j, side in enumerate(PANO_SIDES):
        headers[side] = {
            "roi": out.name,
            "line": n,
            "street_line": pick.get("street_line"),
            "sequence": pick["sequence"],
            "role": "rig sibling" if j else "primary",
            "camera": pick["camera"],
            "view_deg": PANO_SIDES[side],
            "date": pick["date"],
            "resolution": "original",
        }
    t0 = time.time()
    _, imgs = download_pano(imgs, folders, headers, view_deg, fov_deg, store=store)
    TIMINGS["download_s"] += time.time() - t0
    TIMINGS["frames"] += len(imgs)
    return [
        {
            "folder": folders[side].name,
            **{
                key: headers[side][key]
                for key in ("sequence", "line", "role", "camera", "view_deg", "date")
            },
            "frames": len(imgs),
            "spacing_m": round(pick["spacing_m"], 2),
            "pano_side": side,
            "line_coverage": round(pick["coverage"], 3) if side == "right" else None,
        }
        for side in PANO_SIDES
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--roi",
        type=Path,
        required=True,
        help="ROI file from route_gui.py (or a ROI folder)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="ROI folder (default: next free outputs/alignment/mapillary_roi_N)",
    )
    parser.add_argument(
        "--view", choices=list(VIEWS), help="Only start from cameras looking this way"
    )
    parser.add_argument(
        "--view-tol",
        type=float,
        default=35.0,
        help="Allowed deviation (deg) for --view",
    )
    parser.add_argument(
        "--sequence", help="Start from this Mapillary sequence instead of the best one"
    )
    parser.add_argument(
        "--min-images", type=int, default=20, help="Ignore runs with fewer frames"
    )
    parser.add_argument(
        "--max-spacing",
        type=float,
        default=10.0,
        help="Prefer runs whose median frame spacing (m) is at most this",
    )
    parser.add_argument(
        "--max-images", type=int, help="Download at most this many frames per sequence"
    )
    parser.add_argument(
        "--no-rig", action="store_true", help="Do not add the rig's other cameras"
    )
    parser.add_argument(
        "--resolution",
        choices=["2048", "original"],
        default="2048",
        help="Perspective frames' size (360 frames are always downloaded at full size)",
    )
    parser.add_argument(
        "--pano",
        choices=["auto", "prefer", "off"],
        default="auto",
        help="360 runs: taken when they cover more of the line (auto), whenever usable (prefer), "
        "or never (off)",
    )
    parser.add_argument(
        "--pano-fov",
        type=float,
        default=90.0,
        help="Field of view of the 360 crops (deg, square)",
    )
    parser.add_argument("--list", action="store_true", help="Only list candidate runs")
    parser.add_argument(
        "--token",
        default=os.environ.get("MAPILLARY_ACCESS_TOKEN")
        or os.environ.get("MAPILLARY_TOKEN"),
    )
    args = parser.parse_args()
    if not args.token:
        raise SystemExit(
            "Set MAPILLARY_ACCESS_TOKEN (client access token from "
            "https://www.mapillary.com/dashboard/developers) or pass --token"
        )
    roi_file = (
        RoiPaths(args.roi).roi_json if args.roi.is_dir() else args.roi
    )  # an ROI folder: its drawn route
    fetch_roi(
        json.loads(roi_file.read_text()),
        args.token,
        args.out,
        args.view,
        args.view_tol,
        args.sequence,
        args.min_images,
        args.max_spacing,
        args.max_images,
        args.no_rig,
        args.resolution,
        args.list,
        pano=args.pano,
        pano_fov=args.pano_fov,
    )


if __name__ == "__main__":
    main()
