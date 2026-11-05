"""The cameras and the alignment's point cloud in the exported model's coordinates.

export_mesh moves the mesh from UTM metres (z up) to the .glb's frame: minus the UTM
origin at the box centre, minus the centre of the model's bounds, then glTF's y up
(x, y, z) -> (x, z, -y). The same transform puts these overlays on the model; the web
app's viewer draws them on request.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CAMERA_HEIGHT_M = 1.6  # texturing.tex_projection.CAMERA_HEIGHT_M


def _glb(points: np.ndarray, origin: np.ndarray, centre: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(points) - origin - centre
    return np.c_[q[:, 0], q[:, 2], -q[:, 1]]


def _glb_dir(d: np.ndarray) -> np.ndarray:
    d = np.atleast_2d(d)
    return np.c_[d[:, 0], d[:, 2], -d[:, 1]]


def build_overlays(
    camera_poses: dict[str, dict] | None,
    points: dict[str, np.ndarray] | None,
    transformer,
    origin_xy,
    centre,
    terrain=None,
) -> dict[str, Any]:
    """{"cameras": [...], "points": {...}} in the .glb's frame. camera_poses: the
    pipeline's camera_poses metadata (align_poses); points: pose_alignment.points'
    aligned_points. transformer: lon/lat -> the mesh's UTM; origin_xy: the UTM origin
    subtracted from the mesh; centre: the centre of the model's bounds after that."""
    from texturing.tex_projection import _build_R

    origin = np.array([origin_xy[0], origin_xy[1], 0.0])
    centre = np.asarray(centre, float)

    def ground(x, y):
        return terrain.ground(x, y) if terrain is not None else np.zeros(np.shape(x))

    cameras = []
    for img_id, meta in (camera_poses or {}).items():
        if meta.get("latitude") is None or (
            meta.get("rotvec") is None and meta.get("yaw") is None
        ):
            continue
        x, y = transformer.transform(meta["longitude"], meta["latitude"])
        h = meta.get("camera_height_m")
        z = float(ground(x, y)) + (CAMERA_HEIGHT_M if h is None else h)
        R = _build_R(meta)  # world (UTM, z up) -> camera (x right, y down, z forward)
        w, hgt = meta.get("original_width"), meta.get("original_height")
        f = meta.get("focal_length_pixels")
        fov = float(np.degrees(2 * np.arctan(hgt / (2 * f)))) if f and hgt else 60.0
        cameras.append(
            {
                "id": str(img_id),
                "source": meta.get("pose_source", "raw"),
                "p": np.round(_glb(np.array([x, y, z]), origin, centre)[0], 3).tolist(),
                "f": np.round(_glb_dir(R[2])[0], 4).tolist(),
                "u": np.round(_glb_dir(-R[1])[0], 4).tolist(),
                "fov": round(fov, 1),
                "aspect": round(w / hgt, 3) if w and hgt else 1.333,
            }
        )
    out: dict[str, Any] = {"cameras": cameras}
    if points is not None and len(points["lon"]):
        x, y = transformer.transform(points["lon"], points["lat"])
        x, y = np.asarray(x), np.asarray(y)
        z = ground(x, y) + points["height_m"]
        xyz = _glb(np.c_[x, y, z], origin, centre)
        out["points"] = {
            "xyz": np.round(xyz.ravel(), 2).tolist(),
            "rgb": np.asarray(points["rgb"], np.uint8).ravel().tolist(),
        }
    return out
