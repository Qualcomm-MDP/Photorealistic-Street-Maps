"""Bake projected photos onto extruded OSM buildings for a web viewer.

Inputs: an export_blender.py scene (aligned pinhole cameras + undistorted images) and the ROI's
OSM buildings. Every PROJECT_EVERY-th frame of each leg camera is a projector. For each wall
texel (TEXEL m), every projector that has the wall in front of it, in view and not blocked by
another building (plan-view ray test) contributes its colour, weighted by how square-on and how
close it sees the texel (and by its segmentation mask: building pixels count fully, sky/vehicles/
people not at all; unreliable legs at half weight); the weights are sharpened so the best view
dominates. Textures are packed into 2048 px JPEG atlas pages.

Writes <out>/data/scene.json (buildings, walls with atlas UVs, cameras, paths, sparse points),
<out>/data/atlas_<k>.jpg, <out>/data/cams/<k>.jpg (each projector's photo, 512 px, for the viewer's
look-through mode) and <out>/index.html (a copy of viewer.html next to this script). Publish <out>
as the page with its data/ files.

Steps after an ROI is aligned (from Alignment_Pipeline/):
  python export_blender.py rois/mapillary_roi_28
  python viewer/bake_projection.py rois/mapillary_roi_28 rois/mapillary_roi_28/results/blender \
      rois/mapillary_roi_28/results/viewer --every 3 \
      --subtitle "Downtown Rochester, NY · GoPro Max 360, left and right views"
Open <out>/index.html through a local web server (python -m http.server -d <out>), since the page
fetches data/scene.json, or publish <out> as a page.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # Alignment_Pipeline/
sys.path.insert(0, str(ROOT))
from common import MASK_BUILDING, MASK_OTHER, MASK_ROAD, MASK_SKY_OR_MOVING, RoiPaths  # noqa: E402
from export_blender import read_colmap_camera, undistort_maps  # noqa: E402
from osm import building_heights  # noqa: E402

PROJECT_EVERY = 5     # default; --every
TEXEL = 0.1           # m per texel
REACH = 60.0          # m: a projector paints walls within this distance
CONTEXT = 160.0       # m: untextured buildings shown for context within this of the route
PAGE = 2048
SHARPEN = 8           # weight exponent: the best view dominates, neighbours soften seams
MASK_W = {MASK_BUILDING: 1.0, MASK_OTHER: 0.35, MASK_ROAD: 0.2, MASK_SKY_OR_MOVING: 0.0}
EARTH_R = 6378137.0
AXIS_FLIP = np.diag([1.0, -1.0, -1.0])
WALL_RGB = np.array([196, 199, 204], np.uint8)  # texels no photo reaches (BGR, neutral)


def ll_to_xy(lat, lon, lat0, lon0):
    return (EARTH_R * math.radians(lon - lon0) * math.cos(math.radians(lat0)), EARTH_R * math.radians(lat - lat0))


def seg_hits(P0, P1, A, B):
    """For segments P0[i]->P1[i] (N,2) and edges A[j]->B[j] (M,2): (N,) any proper crossing."""
    if len(A) == 0:
        return np.zeros(len(P0), bool)
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    p0, p1 = P0[:, None], P1[:, None]
    a, b = A[None], B[None]
    d1, d2 = orient(a, b, p0), orient(a, b, p1)
    d3, d4 = orient(p0, p1, a), orient(p0, p1, b)
    return ((d1 * d2 < 0) & (d3 * d4 < 0)).any(axis=1)


def main(roi_dir, export_dir, out, every=PROJECT_EVERY, title=None, subtitle=""):
    t0 = time.time()
    p = RoiPaths(roi_dir)
    scene = json.loads((Path(export_dir) / "scene.json").read_text())
    lat0, lon0 = scene["ref"]["lat0"], scene["ref"]["lon0"]
    (out / "data").mkdir(parents=True, exist_ok=True)

    # ---- cameras
    reliable = {}
    for e in json.loads(p.summary_json.read_text())["legs"]:
        reliable[f"{e['leg']}_{e['folder']}"] = bool(e.get("reliable"))
    seqs = {}
    for im in scene["images"]:
        seqs.setdefault(im["folder"], []).append(im)
    sides = {s["folder"]: (s.get("pano_side") or ("right" if (s.get("view_deg") or 0) > 0 else "left")) for s in p.sequences()}
    cams, paths = [], []
    for key, ims in sorted(seqs.items()):
        ims.sort(key=lambda im: im["frame"])
        leg, folder = key.split("_", 1)
        tf = json.loads((p.alignment / key / "transform.json").read_text())
        cmodel, w, h, params = read_colmap_camera(Path(tf["model"]))
        mx, my = undistort_maps(cmodel, w, h, params)
        xyz = []
        for k, im in enumerate(ims):
            a = im["aligned"]
            x, y = ll_to_xy(a["lat"], a["lon"], lat0, lon0)
            C = np.array([x, y, im["camera_height_m"]])
            xyz.append(C)
            if k % every:
                continue
            R = np.array(a["matrix"]) @ AXIS_FLIP  # OpenCV camera -> ENU (columns: right, down, forward)
            cams.append({"key": key, "leg": leg, "folder": folder, "frame": im["frame"], "C": C, "R": R,
                         "f": a["focal_px"], "w": im["width"], "h": im["height"],
                         "image": Path(export_dir) / a["image"], "mask": p.masks / folder / f"{Path(im['frame']).stem}.png",
                         "maps": (mx, my), "reliable": reliable.get(key, False), "side": sides.get(folder, "?")})
        paths.append({"key": key, "leg": leg, "side": sides.get(folder, "?"), "reliable": reliable.get(key, False),
                      "xyz": np.round(np.array(xyz), 2).tolist()})
    camC = np.array([c["C"] for c in cams])
    route = np.vstack([np.array(pth["xyz"])[:, :2] for pth in paths])
    print(f"{len(cams)} projectors (every {every} frames of {len(seqs)} leg cameras)")

    # ---- buildings
    raw = json.loads((p.alignment / "osm_buildings.json").read_text())
    blds = []
    for tags, ring in raw:
        xy = np.array([ll_to_xy(lat, lon, lat0, lon0) for lon, lat in ring])
        if len(xy) < 4:
            continue
        dmin = np.min(np.linalg.norm(route[::5, None] - xy[None], axis=2))
        if dmin > CONTEXT:
            continue
        base, top, _ = building_heights(tags, 3.5, 3)
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
        blds.append({"xy": xy, "base": float(base), "top": float(max(top, base + 2.5)), "ccw": area > 0, "dmin": dmin})
    A = np.vstack([b["xy"] for b in blds]); B = np.vstack([np.roll(b["xy"], -1, axis=0) for b in blds])
    print(f"{len(blds)} buildings within {CONTEXT:.0f} m")
    # Walls of two OSM outlines lying on top of each other (a wall shared by touching buildings, a building
    # part over its building): drawn twice they flicker (z-fighting). The taller building's wall covers the
    # other, so the lower one's edge is left out where the overlap covers >= 80% of it
    edge_of = [(bi, j) for bi, b in enumerate(blds) for j in range(len(b["xy"]))]
    skip = set()
    for i, (bi, j) in enumerate(edge_of):
        a, c = A[i], B[i]
        L = np.linalg.norm(c - a)
        if L < 0.5:
            continue
        u = (c - a) / L; nv = np.array([-u[1], u[0]])
        da, dc = (A - a) @ nv, (B - a) @ nv
        cand = np.flatnonzero((np.abs(da) < 0.15) & (np.abs(dc) < 0.15))
        for k in cand:
            bk, jk = edge_of[k]
            if bk == bi:
                continue
            s0, s1 = sorted([(A[k] - a) @ u, (B[k] - a) @ u])
            if min(L, s1) - max(0.0, s0) >= 0.8 * L:  # the other edge covers this one
                if (blds[bk]["top"], -bk) > (blds[bi]["top"], -bi):
                    skip.add((bi, j))
    print(f"{len(skip)} wall edges left out where outlines overlap")

    imgs, masks = {}, {}

    def load(c):
        if c["frame"] + c["key"] not in imgs:
            imgs[c["frame"] + c["key"]] = cv2.imread(str(c["image"]))
            m = cv2.imread(str(c["mask"]), cv2.IMREAD_GRAYSCALE)
            masks[c["frame"] + c["key"]] = (cv2.remap(m, *c["maps"], cv2.INTER_NEAREST) if m is not None else None)
        return imgs[c["frame"] + c["key"]], masks[c["frame"] + c["key"]]

    # ---- bake walls
    walls, textures, seen_total, texel_total = [], [], 0, 0
    for bi, b in enumerate(blds):
        if b["dmin"] > REACH:
            continue
        n_xy = len(b["xy"])
        for j in range(n_xy):
            if (bi, j) in skip:
                continue
            a2, b2 = b["xy"][j], b["xy"][(j + 1) % n_xy]
            L = float(np.linalg.norm(b2 - a2))
            if L < 1.0:
                continue
            d = (b2 - a2) / L
            nrm = np.array([d[1], -d[0]]) if b["ccw"] else np.array([-d[1], d[0]])  # outward
            mid = (a2 + b2) / 2
            to_cam = camC[:, :2] - mid
            dist = np.linalg.norm(to_cam, axis=1)
            cand = np.flatnonzero((dist < REACH) & (to_cam @ nrm > 0.5))
            if not len(cand):
                continue
            H_m = b["top"] - b["base"]
            texel = max(TEXEL, L / PAGE, H_m / (PAGE / 2))
            W, Hh = max(2, int(math.ceil(L / texel))), max(2, int(math.ceil(H_m / texel)))
            s = (np.arange(W) + 0.5) / W
            colxy = a2[None] + s[:, None] * (b2 - a2)[None]                       # (W, 2)
            z = b["top"] - (np.arange(Hh) + 0.5) / Hh * H_m                       # (Hh,)
            P = np.concatenate([np.broadcast_to(colxy[None], (Hh, W, 2)), np.broadcast_to(z[:, None, None], (Hh, W, 1))], axis=2)
            acc, wsum = np.zeros((Hh, W, 3)), np.zeros((Hh, W))
            near_edges = None
            for ci in cand:
                c = cams[ci]
                # occlusion by other buildings (plan view), per texel column
                lo = np.minimum(colxy.min(0), c["C"][:2]) - 1; hi = np.maximum(colxy.max(0), c["C"][:2]) + 1
                sel = np.all((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo), axis=1)
                own = np.all(np.isclose(A[sel], a2), axis=1) & np.all(np.isclose(B[sel], b2), axis=1)
                ends = colxy - (colxy - c["C"][:2]) * (0.3 / np.maximum(np.linalg.norm(colxy - c["C"][:2], axis=1), 1e-6))[:, None]
                blocked = seg_hits(np.broadcast_to(c["C"][:2], colxy.shape), ends, A[sel][~own], B[sel][~own])
                if blocked.all():
                    continue
                Xc = (P - c["C"]) @ c["R"]                                       # camera frame
                zc = Xc[..., 2]
                ok = zc > 0.5
                u = c["f"] * Xc[..., 0] / np.where(ok, zc, 1) + c["w"] / 2
                v = c["f"] * Xc[..., 1] / np.where(ok, zc, 1) + c["h"] / 2
                ok &= (u >= 0) & (u < c["w"] - 1) & (v >= 0) & (v < c["h"] - 1) & ~blocked[None, :]
                if not ok.any():
                    continue
                ray = c["C"] - P
                rn = np.linalg.norm(ray, axis=2)
                cos = np.clip((ray[..., :2] @ nrm) / rn, 0, 1)
                img, msk = load(c)
                mw = np.ones_like(cos)
                if msk is not None:
                    mv = cv2.remap(msk, u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST)
                    mw = np.select([mv == v for v in MASK_W], list(MASK_W.values()), 0.35)
                wgt = (cos * mw / (1 + rn / 15.0)) ** SHARPEN * (1.0 if c["reliable"] else 0.5 ** SHARPEN) * ok
                if not wgt.any():
                    continue
                col = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
                acc += col * wgt[..., None]
                wsum += wgt
            seen = wsum > 1e-12
            if not seen.any():
                continue
            tex = np.empty((Hh, W, 3), np.uint8)
            tex[:] = WALL_RGB
            tex[seen] = np.clip(acc[seen] / wsum[seen][:, None], 0, 255).astype(np.uint8)
            seen_total += int(seen.sum()); texel_total += seen.size
            walls.append({"b": bi, "j": j, "a": [round(float(v), 3) for v in a2], "c": [round(float(v), 3) for v in b2],
                          "base": b["base"], "top": b["top"], "W": W, "H": Hh, "seen": round(float(seen.mean()), 3)})
            textures.append(tex)
        if bi % 25 == 0:
            print(f"  building {bi}/{len(blds)}: {len(walls)} textured walls, {time.time() - t0:.0f} s", flush=True)

    # ---- pack into atlas pages (shelf packing, tallest first)
    order = sorted(range(len(walls)), key=lambda k: -walls[k]["H"])
    pages, cur, x, y, shelf = [], None, 0, 0, 0
    def new_page():
        img = np.empty((PAGE, PAGE, 3), np.uint8); img[:] = WALL_RGB
        pages.append(img)
        return img
    cur = new_page()
    for k in order:
        wl, t = walls[k], textures[k]
        Wt, Ht = t.shape[1] + 2, t.shape[0] + 2
        if x + Wt > PAGE:
            x, y, shelf = 0, y + shelf, 0
        if y + Ht > PAGE:
            cur, x, y, shelf = new_page(), 0, 0, 0
        cur[y + 1:y + 1 + t.shape[0], x + 1:x + 1 + t.shape[1]] = t
        wl.update(page=len(pages) - 1, uv=[round((x + 1) / PAGE, 6), round(1 - (y + 1) / PAGE, 6),
                                           round((x + 1 + t.shape[1]) / PAGE, 6), round(1 - (y + 1 + t.shape[0]) / PAGE, 6)])
        x += Wt; shelf = max(shelf, Ht)
    for i, pg in enumerate(pages):
        cv2.imwrite(str(out / "data" / f"atlas_{i}.jpg"), pg, [cv2.IMWRITE_JPEG_QUALITY, 82])

    # ---- sparse points (subsample)
    pts = None
    ply = Path(export_dir) / "points.ply"
    if ply.exists():
        raw_ply = ply.read_bytes()
        head, body = raw_ply.split(b"end_header\n", 1)
        n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        v = np.frombuffer(body[:n * 15], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rng = np.random.default_rng(0)
        idx = rng.choice(n, min(n, 40000), replace=False)
        sp = scene["points"]
        dx, dy = ll_to_xy(sp["lat0"], sp["lon0"], lat0, lon0)
        pts = {"xyz": np.round(np.c_[v["x"][idx] + dx, v["y"][idx] + dy, v["z"][idx]], 2).ravel().tolist(),
               "rgb": np.c_[v["r"][idx], v["g"][idx], v["b"][idx]].ravel().tolist()}

    # each projector's photo, small, for the viewer's look-through mode
    (out / "data" / "cams").mkdir(exist_ok=True)
    for i, c in enumerate(cams):
        img = cv2.imread(str(c["image"]))
        img = cv2.resize(img, (512, round(512 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "data" / "cams" / f"{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    viewer = Path(__file__).with_name("viewer.html")
    if viewer.exists():
        shutil.copy2(viewer, out / "index.html")

    doc = {"roi": p.name, "title": title or f"{p.name} · projected facades", "subtitle": subtitle, "ref": {"lat0": lat0, "lon0": lon0}, "project_every": every, "texel_m": TEXEL,
           "pages": [f"data/atlas_{i}.jpg" for i in range(len(pages))],
           "buildings": [{"xy": np.round(b["xy"], 2).tolist(), "base": b["base"], "top": b["top"],
                          "near": bool(b["dmin"] <= REACH),
                          "skip": sorted(j for (bk, j) in skip if bk == bi)} for bi, b in enumerate(blds)],
           "walls": walls,
           "projectors": [{"key": c["key"], "leg": c["leg"], "side": c["side"], "frame": c["frame"], "reliable": c["reliable"],
                           "C": np.round(c["C"], 2).tolist(), "R": np.round(c["R"], 5).tolist(), "f": c["f"],
                           "w": c["w"], "h": c["h"], "photo": f"data/cams/{i:03d}.jpg"} for i, c in enumerate(cams)],
           "paths": paths, "points": pts,
           "stats": {"walls_textured": len(walls), "coverage": round(seen_total / max(texel_total, 1), 3),
                     "pages": len(pages), "projectors": len(cams), "bake_s": round(time.time() - t0)}}
    (out / "data" / "scene.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(json.dumps(doc["stats"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roi_dir", type=Path)
    ap.add_argument("export_dir", type=Path, help="export_blender.py --out folder")
    ap.add_argument("out", type=Path)
    ap.add_argument("--every", type=int, default=PROJECT_EVERY, help="Project every n-th frame of each leg camera")
    ap.add_argument("--title", help="Page heading (default: '<roi> · projected facades')")
    ap.add_argument("--subtitle", default="", help="Line under the heading, e.g. the place and camera")
    a = ap.parse_args()
    main(a.roi_dir, a.export_dir, a.out, a.every, a.title, a.subtitle)
