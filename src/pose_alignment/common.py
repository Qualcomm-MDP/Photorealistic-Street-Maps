"""Shared helpers for the pipeline layers: HTTP and Overpass, a local metric frame, the ROI
folder layout, COLMAP model readers, 2D similarity transforms and the segmentation mask
values."""

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROIS_DIR = HERE / "rois"      # one parent folder per ROI: rois/mapillary_roi_N/{data,results}
ROUTES_DIR = HERE / "routes"  # routes drawn in route_gui.py
USER_AGENT = "umich-mdp-alignment-pipeline/1.0"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]

# Values in the segmentation masks (segmentation_layer.py). Spread over 0-255 so a mask PNG
# is readable in any image viewer: black other, dark grey road, light grey building, white
# sky and things that move.
MASK_OTHER, MASK_ROAD, MASK_BUILDING, MASK_SKY_OR_MOVING = 0, 100, 200, 255
MASK_NAMES = {MASK_OTHER: "other", MASK_ROAD: "road", MASK_BUILDING: "building",
              MASK_SKY_OR_MOVING: "sky_or_moving"}


# ----------------------------------------------------------------------------- web

def get_json(url, data=None, headers=None, retries=3):
    """GET/POST and parse JSON, retrying transient failures (429, 5xx, timeouts)."""
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == retries - 1:
                raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}: {e.read()[:300].decode(errors='replace')}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"{url.split('?')[0]}: {e}")
        time.sleep(5 * (attempt + 1))


def overpass(query):
    """Run an Overpass QL query, trying each server in turn."""
    for url in OVERPASS_URLS:
        try:
            return get_json(url, data=urllib.parse.urlencode({"data": query}).encode())
        except RuntimeError as e:
            print(f"  Overpass failed ({str(e)[:80]}), trying next server")
    raise SystemExit("All Overpass servers failed; try again later")


# ----------------------------------------------------------------------------- geography

class LocalFrame:
    """Equirectangular metres (x east, y north) around a reference point; fine over a few km."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = 111320.0 * math.cos(math.radians(lat0))
        self.ky = 110540.0

    def xy(self, lon, lat):
        return ((lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky)

    def lonlat(self, x, y):
        return (self.lon0 + x / self.kx, self.lat0 + y / self.ky)


def line_length_m(line):
    """Length (m) of a [(lon, lat), ...] polyline."""
    frame = LocalFrame(*line[0])
    pts = [frame.xy(*p) for p in line]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def heading_deg(a, b):
    """Compass heading (deg clockwise from north) from point a to point b, both (x east, y north)."""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))


def straight_stretches(xy, times_ms, max_turn_deg=35.0, trim_deg=20.0, min_spacing=1.0, max_gap_s=300.0,
                       max_jump_m=40.0):
    """Straight stretches of a camera path in capture order: lists of indices into xy.

    The path is split at time gaps, position jumps and sharp turns, and each piece is
    trimmed to its longest stretch heading within trim_deg of its median heading (which
    drops the curved frames at a corner). Frames closer than min_spacing to the previous
    kept one (stopped) are skipped. A turn is detected when the direction of the last few
    frames (>= 3 steps) differs by more than max_turn_deg from the piece's course so far
    (its start to a few frames ago, once that is >= 10 m, so GPS jitter on short steps
    does not trigger it). Used by both the Mapillary layer (to judge runs) and the COLMAP
    layer (to plan legs), so a run that is picked can be reconstructed."""
    pieces, last, pts = [], None, []
    for i, (p, t) in enumerate(zip(xy, times_ms)):
        new = last is None
        if last is not None:
            step = math.dist(p, xy[last])
            if step < min_spacing:
                continue
            if (t - times_ms[last]) / 1000 > max_gap_s or step > max_jump_m:
                new = True
            elif len(pts) >= 4 and math.dist(pts[0], pts[-3]) >= 10:
                new = abs(wrap_deg(heading_deg(pts[-3], p) - heading_deg(pts[0], pts[-3]))) > max_turn_deg
        if new:
            pieces.append([])
            pts = []
        pieces[-1].append(i)
        pts.append(p)
        last = i
    return [trimmed for piece in pieces if (trimmed := _trim_straight([xy[i] for i in piece], piece, trim_deg))]


def _trim_straight(pts, idx, trim_deg, k=2):
    """The longest stretch of idx whose local heading is within trim_deg of the median."""
    pts = np.asarray(pts, float)
    h = []
    for i in range(len(pts)):
        a, b = pts[max(i - k, 0)], pts[min(i + k, len(pts) - 1)]
        h.append(math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) if np.linalg.norm(b - a) > 0.5 else np.nan)
    h = np.array(h)
    ok = ~np.isnan(h)
    if not ok.any():
        return []
    ref = math.degrees(math.atan2(np.nanmean(np.sin(np.radians(h))), np.nanmean(np.cos(np.radians(h)))))
    good = ok & (np.abs((h - ref + 180) % 360 - 180) <= trim_deg)
    best, cur = (0, 0), None
    for i, g in enumerate(list(good) + [False]):
        if g and cur is None:
            cur = i
        elif not g and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    return idx[best[0]:best[1]]


# ----------------------------------------------------------------------------- ROI layout

class RoiPaths:
    """Folders of one ROI. Everything for an ROI lives in its own parent folder,
    rois/mapillary_roi_N/ by default:

      data/
        mapillary/     roi.json, sequences.json, one folder per sequence
                       (frame_0001.jpg ... and cameras.json)
        masks/         segmentation masks, one folder per sequence
        colmap/        COLMAP models per leg and sequence, legs.json
        alignment/     corrected cameras per sequence (<sequence folder>.json),
                       summary.json and per-leg alignment data
      results/         images: alignment_overview.png, raw_gps_vs_pipeline.png,
                       legs/<leg>_alignment.png, sparse_views/<leg>_<sequence folder>.png
    """

    def __init__(self, roi_dir):
        root = Path(roi_dir).resolve()
        # Accept a subfolder too (e.g. .../mapillary_roi_1/data or its data/mapillary).
        for _ in range(2):
            if root.name in ("data", "mapillary", "results") and not (root / "data").exists():
                root = root.parent
        self.root = root
        self.name = root.name
        self.data = root / "data"
        self.images = self.data / "mapillary"
        self.masks = self.data / "masks"
        self.colmap = self.data / "colmap"
        self.alignment = self.data / "alignment"
        self.results = root / "results"
        self.roi_json = self.images / "roi.json"
        self.sequences_json = self.images / "sequences.json"
        self.legs_json = self.colmap / "legs.json"
        self.summary_json = self.alignment / "summary.json"

    def sequences(self):
        if not self.sequences_json.exists():
            raise SystemExit(f"{self.sequences_json} not found: run the mapillary stage first")
        return json.loads(self.sequences_json.read_text())

    def frames(self, folder):
        """Per-frame Mapillary data of one sequence folder, by file name."""
        return {f["file"]: f for f in json.loads((self.images / folder / "cameras.json").read_text())["frames"]}


def next_roi_dir(rois_dir=ROIS_DIR):
    """rois_dir/mapillary_roi_1, or _2, _3, ... if taken."""
    k = 1
    while (Path(rois_dir) / f"mapillary_roi_{k}").exists():
        k += 1
    return Path(rois_dir) / f"mapillary_roi_{k}"


# ----------------------------------------------------------------------------- COLMAP models

def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_cameras(model_txt):
    """(names, centres Nx3, world-from-camera rotations Nx3x3) from images.txt, sorted by name."""
    lines = [l for l in (Path(model_txt) / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    cams = []
    for line in lines[0::2]:
        p = line.split()
        R = quat_to_rot(*map(float, p[1:5]))  # camera-from-world
        t = np.array(list(map(float, p[5:8])))
        cams.append((p[9], -R.T @ t, R.T))
    cams.sort(key=lambda c: c[0])
    names, centres, rots = zip(*cams)
    return list(names), np.array(centres), np.array(rots)


def load_points(model_txt):
    """(xyz Nx3, rgb Nx3 in 0..1, track length N) from points3D.txt."""
    rows = [l.split() for l in (Path(model_txt) / "points3D.txt").read_text().splitlines() if l and not l.startswith("#")]
    if not rows:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int)
    data = np.array([r[:8] for r in rows], dtype=float)
    track = np.array([(len(r) - 8) // 2 for r in rows])
    return data[:, 1:4], data[:, 4:7] / 255.0, track


def read_camera(model_txt):
    """(model name, width, height, K, OpenCV distortion [k1, k2, p1, p2]) of the model's single camera."""
    line = next(l for l in (Path(model_txt) / "cameras.txt").read_text().splitlines() if l and not l.startswith("#"))
    p = line.split()
    model, w, h, params = p[1], int(p[2]), int(p[3]), [float(v) for v in p[4:]]
    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        return model, w, h, np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([k, 0, 0, 0])
    if model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        return model, w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array([k1, k2, p1, p2])
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return model, w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.zeros(4)
    raise ValueError(f"Unsupported camera model {model}")


def model_size(model_dir):
    """Registered frames in a COLMAP model folder (with a txt/ export)."""
    images = Path(model_dir) / "txt" / "images.txt"
    if not images.exists():
        return 0
    return sum(1 for l in images.read_text().splitlines() if l and not l.startswith("#")) // 2


def gravity_frame(centres, rots):
    """(origin, basis) mapping COLMAP world to (right, forward, up): p = (X - origin) @ basis.T.

    forward = first to last camera; up = perpendicular to forward and to the mean camera
    x (image-right) axis, which stays horizontal without roll. Assumes a straight path."""
    forward = centres[-1] - centres[0]
    forward /= np.linalg.norm(forward)
    up = np.cross(rots[:, :, 0].mean(axis=0), forward)
    up /= np.linalg.norm(up)
    if up.dot(-rots[:, :, 1].mean(axis=0)) < 0:
        up = -up
    right = np.cross(forward, up)
    return centres[0], np.stack([right, forward, up])


def label_points(model_txt, mask_dir, values, min_fraction=0.5):
    """Per 3D point (points3D.txt order): (has the label, observations checked).

    A point has the label when at least min_fraction of its observations fall on mask pixels
    whose value is in `values` (segmentation_layer.py masks, <mask_dir>/<frame stem>.png)."""
    import cv2

    model_txt, mask_dir = Path(model_txt), Path(mask_dir)
    order = [int(l.split()[0]) for l in (model_txt / "points3D.txt").read_text().splitlines()
             if l and not l.startswith("#")]
    index = {pid: i for i, pid in enumerate(order)}
    hits, total = np.zeros(len(order)), np.zeros(len(order))
    values = np.atleast_1d(values)
    lines = [l for l in (model_txt / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for header, obs in zip(lines[0::2], lines[1::2]):
        mask = cv2.imread(str(mask_dir / f"{Path(header.split()[9]).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        v = np.array(obs.split(), float).reshape(-1, 3)
        v = v[v[:, 2] >= 0]
        h, w = mask.shape
        x = np.clip(np.round(v[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(v[:, 1]).astype(int), 0, h - 1)
        rows = np.array([index[int(p)] for p in v[:, 2]], int)
        np.add.at(total, rows, 1)
        np.add.at(hits, rows, np.isin(mask[y, x], values))
    return (total > 0) & (hits / np.maximum(total, 1) >= min_fraction), total


# ----------------------------------------------------------------------------- 2D similarity

def similarity_2d(src, dst):
    """Umeyama in 2D: (scale, angle, translation) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(axis=0).sum()
    return s, math.atan2(R[1, 0], R[0, 0]), mu_d - s * R @ mu_s


def apply_2d(params, xy):
    """Apply [angle, log scale, tx, ty] to (N, 2) points."""
    angle, log_s, tx, ty = params
    c, s = math.cos(angle), math.sin(angle)
    return math.exp(log_s) * xy @ np.array([[c, s], [-s, c]]) + [tx, ty]


# ----------------------------------------------------------------------------- geometry

def segment_distances(P, A, B):
    """Distance from each point (N, 2) to each segment A->B (M, 2): (N, M)."""
    d = B - A
    t = np.clip(np.einsum("nmk,mk->nm", P[:, None] - A, d) / np.maximum((d * d).sum(axis=1), 1e-12), 0, 1)
    return np.linalg.norm(P[:, None] - (A + t[..., None] * d), axis=2)


def fit_plane(P, thresh, rng, iters=1000):
    """RANSAC plane, refined by least squares on its inliers: (unit normal, inlier mask)."""
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        inl = np.abs((P - a) @ (n / np.linalg.norm(n))) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        return None, None
    Q = P[best]
    return np.linalg.svd(Q - Q.mean(axis=0))[2][2], best


def ransac_lines(xy, thresh, min_inliers, iters=2000, rng=None):
    """Sequential RANSAC for 2D lines: [(centre, direction, inlier indices)]."""
    rng = rng or np.random.default_rng(0)
    remaining = np.arange(len(xy))
    lines = []
    while len(remaining) >= min_inliers:
        pts = xy[remaining]
        best = None
        for _ in range(iters):
            a, b = pts[rng.choice(len(pts), 2, replace=False)]
            d = b - a
            if np.linalg.norm(d) < 1e-6:
                continue
            inl = np.flatnonzero(np.abs((pts - a) @ (np.array([-d[1], d[0]]) / np.linalg.norm(d))) < thresh)
            if best is None or len(inl) > len(best):
                best = inl
        if best is None or len(best) < min_inliers:
            break
        # Refine with PCA on the inliers, then recollect inliers against the refined line.
        for _ in range(2):
            c = pts[best].mean(axis=0)
            d = np.linalg.svd(pts[best] - c)[2][0]
            best = np.flatnonzero(np.abs((pts - c) @ np.array([-d[1], d[0]])) < thresh)
        lines.append((c, d, remaining[best]))
        remaining = np.delete(remaining, best)
    return lines


def split_segments(s, max_gap, min_inliers, min_length):
    """Split positions along a line into contiguous runs; return index arrays into s."""
    order = np.argsort(s)
    runs = np.split(order, np.flatnonzero(np.diff(s[order]) > max_gap) + 1)
    return [run for run in runs if len(run) >= min_inliers and np.ptp(s[run]) >= min_length]


def planarity(xy, c, d, s0, s1, thresh):
    """Fraction of nearby points (within 3x thresh) that lie within thresh of the line.
    A wall is a thin sheet, so nearly all nearby points sit on it; a tree canopy is a
    volume, so a thin slice through it holds only about a third of its neighbours."""
    s, dist = (xy - c) @ d, np.abs((xy - c) @ np.array([-d[1], d[0]]))
    span = (s >= s0) & (s <= s1)
    return (span & (dist < thresh)).sum() / max(1, (span & (dist < 3 * thresh)).sum())
