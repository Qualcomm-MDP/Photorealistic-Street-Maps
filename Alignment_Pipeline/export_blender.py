"""Export the aligned cameras as a scene for compare_cameras_blender.py (the repo root's Blender
loader), in the scene.json layout of export_comparison.py and mast3r/export_blender_mast3r.py.

Pose sets per frame:
  aligned    the pipeline's pose: the COLMAP camera levelled with GeoCalib's gravity and placed by
             its leg's (or piece's) OSM fit; full rotation (incl. pitch, roll), the leg's refined
             focal, its height above the road. Background photo: the frame
             undistorted to a pinhole with that leg's refined calibration (Blender cameras have
             no lens distortion), same size and focal, principal point at the centre
  gps        Mapillary's raw GPS fix, camera level and turned to the raw compass heading,
             Mapillary's focal; same photo
Heights are above the road under the path, since the Blender scene is flat: the camera path
lowered by the camera height measured from the road points (streets climb and dip: main_street_east
rises ~2.5 m over 90 m, roi_9 dips ~7 m, both confirmed by Mapillary's altitudes); each point is
taken relative to the ground under its nearest camera. Every aligned leg is its own
sub-collection ("<leg>_<sequence folder>"), so neighbouring legs' cameras can be compared or
hidden separately.

Writes to --out (default results[/<tag>]/blender/):
  scene.json             per frame and leg: id, frame, folder, captured_at, camera_height_m, poses
  images/<leg>_<id>.jpg  undistorted frames
  points.ply             every leg's sparse points placed the same way: east/north metres from
                         the anchor in scene.json ("points"), height above the road under
                         its nearest camera

In Blender: in compare_cameras_blender.py set SCENE_DIR to --out and
LOAD = ("pipeline", "gps", "points"), then run it in the .blend with the Blosm scene.

Usage: python export_blender.py ../Alignment_Pipeline/rois/main_street_east [--tag lg]
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from align import point_pieces
from scipy.ndimage import median_filter

from common import MASK_ROAD, LocalFrame, RoiPaths, label_points, load_cameras, load_points

AXIS_FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV camera (+z forward, +y down) -> Blender (-z forward, +y up)
EARTH_R = 6378137.0  # compare_cameras_blender.latlon_to_xy
SENSOR_W = 36.0
DEFAULT_HEIGHT = 1.5  # metres, camera above the road when the road is not reconstructed well enough
ROAD_RADIUS = 15.0  # metres around a camera for its road points (the road is seen a few metres ahead)


def blender_matrix(R_cam):
    return [[round(float(v), 8) for v in row] for row in R_cam @ AXIS_FLIP]


def heading_pitch(R_cam):
    fwd = R_cam[:, 2]
    return math.degrees(math.atan2(fwd[0], fwd[1])) % 360, math.degrees(math.asin(np.clip(fwd[2], -1, 1)))


def level_camera(heading_deg):
    h = math.radians(heading_deg)
    return np.column_stack([[math.cos(h), -math.sin(h), 0.0], [0.0, 0.0, -1.0], [math.sin(h), math.cos(h), 0.0]])


def read_colmap_camera(model_txt):
    line = next(l for l in (Path(model_txt) / "cameras.txt").read_text().splitlines() if l and not l.startswith("#"))
    p = line.split()
    return p[1], int(p[2]), int(p[3]), [float(v) for v in p[4:]]


def undistort_maps(model, w, h, params):
    """cv2.remap maps from a pinhole (same focal, centred principal point) to the COLMAP camera."""
    f, cx, cy, k1 = params[:4]
    k2 = params[4] if len(params) > 4 else 0.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x, y = (u - w / 2) / f, (v - h / 2) / f
    r = np.hypot(x, y)
    if model == "RADIAL_FISHEYE":  # theta_d = theta (1 + k1 theta^2 + k2 theta^4), as COLMAP
        th = np.arctan(r)
        scale = np.where(r > 1e-9, th * (1 + k1 * th ** 2 + k2 * th ** 4) / np.maximum(r, 1e-9), 1.0)
    else:  # SIMPLE_RADIAL / RADIAL
        r2 = r ** 2
        scale = 1 + k1 * r2 + k2 * r2 ** 2
    return (f * x * scale + cx).astype(np.float32), (f * y * scale + cy).astype(np.float32)


def write_ply(path, xyz, rgb):
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["r"], v["g"], v["b"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(v.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    p = RoiPaths(args.roi_dir, args.tag)
    out = (args.out or p.results / "blender").resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)

    images, pts_ll, pts_z, pts_rgb, heights = [], [], [], [], []
    for d in sorted(x for x in p.alignment.iterdir() if (x / "transform.json").exists()):
        tf = json.loads((d / "transform.json").read_text())
        stats = json.loads((d / "stats.json").read_text())
        leg, folder = stats["leg"], stats["folder"]
        model = Path(tf["model"])
        meta = p.frames(folder)
        enu = LocalFrame(*tf["enu_origin_lon_lat"])
        origin, basis, ground_z = np.array(tf["origin"]), np.array(tf["basis"]), tf["ground_z"]
        pieces = tf["pieces"]
        names, C, R = load_cameras(model)
        piece_of = {n: k for k, pc in enumerate(pieces) for n in pc["images"]}
        cam_piece = np.array([piece_of[n] for n in names])

        def place(X, idx):
            """ENU east/north and height above ground; R_z per row."""
            L = (X - origin) @ basis.T
            xy, z, Rz = np.empty((len(X), 2)), np.empty(len(X)), np.empty((len(X), 3, 3))
            for k, pc in enumerate(pieces):
                m = idx == k
                s, a, t = pc["scale_m_per_unit"], pc["angle_rad"], np.array(pc["translation_m"])
                rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                xy[m] = s * L[m, :2] @ rz[:2, :2].T + t
                z[m] = s * (L[m, 2] - ground_z)
                Rz[m] = rz
            return xy, z, Rz

        xyz, rgb, track = load_points(model)
        pt_piece = point_pieces(model, names, piece_of) if len(pieces) > 1 else np.zeros(len(xyz), int)
        pxy, pz, _ = place(xyz, pt_piece)
        cxy, cz, cRz = place(C, cam_piece)
        # Ground under the path: the camera path itself lowered by the camera height, as the vehicle
        # follows the road wherever it climbs or dips (main_street_east climbs 1.5 deg; roi_9 dips
        # ~7 m under a footbridge, so no single plane fits). Camera height: median over cameras of
        # the height above the road points within ROAD_RADIUS below it; DEFAULT_HEIGHT where the road is
        # barely reconstructed or the result is implausible
        road, _ = label_points(model, p.masks / folder, MASK_ROAD, 0.5)
        hs = []
        for i in range(len(names)):
            m = road & (np.hypot(*(pxy - cxy[i]).T) < ROAD_RADIUS) & (pz < cz[i]) & (pz > cz[i] - 4)
            if m.sum() >= 5:
                hs.append(cz[i] - np.median(pz[m]))
        cam_h = float(np.median(hs)) if len(hs) >= len(names) / 3 else float("nan")
        source = f"road points under {len(hs)} cameras"
        if not 0.5 <= cam_h <= 3.0:
            cam_h, source = DEFAULT_HEIGHT, f"default (road points gave {cam_h:.2f} m)"
        ground = median_filter(cz, size=5, mode="nearest") - cam_h
        nearest = np.argmin(np.hypot(pxy[:, None, 0] - cxy[None, :, 0], pxy[:, None, 1] - cxy[None, :, 1]), axis=1)
        cz, pz = cz - ground, pz - ground[nearest]
        cmodel, w, h, params = read_colmap_camera(model)
        f_px = params[0]
        mx, my = undistort_maps(cmodel, w, h, params)
        for i, n in enumerate(names):
            m = meta[n]
            image_id = str(m["id"])
            img_name = f"images/{leg}_{image_id}.jpg"
            src = cv2.imread(str(p.images / folder / n))
            if src.shape[1] != w:  # COLMAP worked on the same frames; guard against resized copies
                src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / img_name), cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            R_cam = cRz[i] @ basis @ R[i]  # OpenCV camera -> ENU
            ha, pa = heading_pitch(R_cam)
            lon, lat = enu.lonlat(*cxy[i])
            heights.append(float(cz[i]))
            gps = None
            if m.get("gps") and m.get("compass_angle") is not None:
                fm = (m.get("camera_parameters") or [0.85])[0] * max(w, h)
                gx, gy = enu.xy(*m["gps"])
                gps = {"lat": m["gps"][1], "lon": m["gps"][0], "matrix": blender_matrix(level_camera(m["compass_angle"])),
                       "lens_mm": fm * SENSOR_W / w, "focal_px": fm, "heading_deg": m["compass_angle"] % 360,
                       "pitch_deg": 0.0, "image": img_name, "offset_m": float(np.hypot(cxy[i, 0] - gx, cxy[i, 1] - gy))}
            mx_, my_ = enu.xy(m["lon"], m["lat"])
            ch = m.get("computed_compass_angle")
            images.append({
                "id": image_id, "frame": n, "folder": f"{leg}_{folder}", "leg": leg, "piece": int(cam_piece[i]),
                "reliable": bool(stats["reliable"]), "camera_height_m": float(cz[i]),
                "captured_at": m.get("captured_at_ms") or int(datetime.fromisoformat(m["captured_at"]).timestamp() * 1000),
                "width": w, "height": h,
                "aligned": {"lat": lat, "lon": lon, "matrix": blender_matrix(R_cam), "lens_mm": f_px * SENSOR_W / w,
                            "focal_px": f_px, "heading_deg": ha, "pitch_deg": pa, "image": img_name},
                "gps": gps,
                "offset_m": float(np.hypot(cxy[i, 0] - mx_, cxy[i, 1] - my_)),
                "heading_diff_deg": float((ha - ch + 180) % 360 - 180) if ch is not None else None,
            })

        keep = (track >= 2) & (pz > -3) & (pz < 40) & (np.hypot(*(pxy - cxy.mean(0)).T) < 150)
        pts_ll.append(np.array([enu.lonlat(x, y) for x, y in pxy[keep]]))
        pts_z.append(pz[keep])
        pts_rgb.append((rgb[keep] * 255).round().astype(np.uint8))
        print(f"{leg}/{folder}: {len(names)} cameras ({len(pieces)} piece{'s' if len(pieces) > 1 else ''}, "
              f"{'reliable' if stats['reliable'] else 'UNRELIABLE'}), {keep.sum()} points, camera height {cam_h:.2f} m "
              f"({source})")

    if not images:
        raise SystemExit(f"No aligned legs in {p.alignment}: run align.py first")
    ll = np.vstack(pts_ll)
    lon0, lat0 = float(ll[:, 0].mean()), float(ll[:, 1].mean())
    px = EARTH_R * np.radians(ll[:, 0] - lon0) * math.cos(math.radians(lat0))
    py = EARTH_R * np.radians(ll[:, 1] - lat0)
    write_ply(out / "points.ply", np.c_[px, py, np.concatenate(pts_z)], np.vstack(pts_rgb))

    off = np.array([im["offset_m"] for im in images])
    goff = np.array([im["gps"]["offset_m"] for im in images if im["gps"]])
    dh = np.array([im["heading_diff_deg"] for im in images if im["heading_diff_deg"] is not None])
    stat = lambda a: {"median": float(np.median(a)), "max": float(a.max()), "min": float(a.min())}  # noqa: E731
    (out / "scene.json").write_text(json.dumps({
        "ref": {"lat0": lat0, "lon0": lon0}, "camera_height_m": float(np.median(heights)),
        "source": str(p.alignment), "label": f"Alignment_Pipeline{f' ({args.tag})' if args.tag else ''}",
        "summary": {"cameras": len(images), "offset_m": stat(off), "gps_offset_m": stat(goff) if len(goff) else None,
                    "heading_diff_deg": {"median": float(np.median(dh)) if len(dh) else float("nan"),
                                         "max_abs": float(np.abs(dh).max()) if len(dh) else float("nan")}},
        "images": images,
        "points": {"file": "points.ply", "count": int(len(px)), "lat0": lat0, "lon0": lon0, "earth_radius_m": EARTH_R,
                   "note": "x east, y north (metres from lat0/lon0, equirectangular), z above ground"},
    }, indent=1))
    print(f"{len(images)} cameras, {len(px):,} points -> {out}")
    print(f'In compare_cameras_blender.py set SCENE_DIR = "{out}" and LOAD = ("pipeline", "gps", "points")')


if __name__ == "__main__":
    main()
