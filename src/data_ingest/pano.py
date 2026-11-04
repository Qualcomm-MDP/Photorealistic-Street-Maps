"""360 photos as pinhole views looking right and left of travel.

A 360 (equirectangular) photo cannot be projected like a perspective photo, so it is cut
into two square pinhole views, ±90° from the direction of travel and fov_deg across
(default 90°): where the facades on either side of the street are. The pose alignment
downloads 360 runs this way (data_ingest.mapillary_sequences.download_pano), and the fetch
stage turns every other 360 photo into the same two views for texturing (pano_views).

The crop directions are fixed in the 360 image: the side's angle minus the sequence's
median view angle (where the image centre looks relative to travel; 0 when unknown).
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

PANO_TYPES = ("spherical", "equirectangular")
# crop directions relative to travel; the first is the primary
PANO_SIDES = {"right": 90.0, "left": -90.0}
PANO_FOV_DEG = 90.0
PANO_MAX_WIDTH = 2048  # crops no larger than a perspective frame at 2048 px

_maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_maps_lock = threading.Lock()  # download_pano crops in threads: each map is built once


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def is_pano(img: dict) -> bool:
    return bool(img.get("is_pano")) or img.get("camera_type") in PANO_TYPES


def crop_axes(yaw_deg: float) -> np.ndarray:
    """Columns: a crop's right, down and forward axes in the 360 camera's frame (x right, y down,
    z the image centre; OpenSfM's spherical camera), for a crop turned yaw_deg to the right."""
    y = math.radians(yaw_deg)
    return np.array(
        [
            [math.cos(y), 0.0, math.sin(y)],
            [0.0, 1.0, 0.0],
            [-math.sin(y), 0.0, math.cos(y)],
        ]
    )


def crop_size(pano_width: float | None, fov_deg: float = PANO_FOV_DEG) -> int:
    """Side (px, even) of a crop with the 360 image's own resolution, at most PANO_MAX_WIDTH."""
    return min(PANO_MAX_WIDTH, int(round((pano_width or 2048) * fov_deg / 360 / 2)) * 2)


def crop_rotvec(rotvec, yaw_deg: float) -> list[float]:
    """A crop's world-to-camera rotation (Rodrigues) from its 360 image's."""
    R = cv2.Rodrigues(np.asarray(rotvec, float))[0]
    return cv2.Rodrigues(crop_axes(yaw_deg).T @ R)[0].ravel().tolist()


def pano_crop_maps(W, H, yaw_deg, fov_deg, size):
    """cv2.remap maps from a size x size pinhole view (fov_deg across, looking yaw_deg right of the
    image centre, level with the 360 image's horizon) into a W x H equirectangular image."""
    f = size / 2 / math.tan(math.radians(fov_deg) / 2)
    u, v = np.meshgrid(
        np.arange(size) - (size - 1) / 2, np.arange(size) - (size - 1) / 2
    )
    rays = np.stack([u / f, v / f, np.ones_like(u)], axis=-1)
    # as one 2-D product: NumPy's stacked (3-D) matmul crashed now and then on macOS
    # (segmentation fault in Accelerate, about 1 call in 10)
    d = (rays.reshape(-1, 3) @ crop_axes(yaw_deg).T).reshape(rays.shape)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arctan2(-d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    return (
        ((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32),
        ((0.5 - lat / math.pi) * H - 0.5).astype(np.float32),
    )


def crop_view(
    pano: np.ndarray, yaw_deg: float, fov_deg: float, size: int
) -> np.ndarray:
    """The size x size pinhole view of a 360 image (BGR) looking yaw_deg right of its centre."""
    # The crop's centre has size/2 px per radian, the 360 image W/2pi: shrink a much sharper 360
    # image first (area averaging), so the remap does not alias
    if pano.shape[1] > 1.5 * math.pi * size:
        pano = cv2.resize(
            pano,
            (int(math.pi * size), int(math.pi * size) // 2),
            interpolation=cv2.INTER_AREA,
        )
    H, W = pano.shape[:2]
    key = (W, H, round(yaw_deg, 6), fov_deg, size)
    with _maps_lock:
        if key not in _maps:
            _maps[key] = pano_crop_maps(W, H, yaw_deg, fov_deg, size)
        mx, my = _maps[key]
    return cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def sequence_view_angles(records: list[dict]) -> dict[str, float | None]:
    """{sequence id: median angle of the 360 image centre relative to travel (deg; None
    when unknown)} over the 360 records, from their compass against their track."""
    from data_ingest.mapillary_sequences import view_angle
    from pose_alignment.geo_common import LocalFrame

    by_seq: dict[str, list[dict]] = {}
    for img in records:
        if is_pano(img) and img.get("sequence"):
            by_seq.setdefault(img["sequence"], []).append(img)
    out = {}
    for seq, imgs in by_seq.items():
        imgs.sort(key=lambda i: i.get("captured_at") or 0)
        lonlat = [i["geometry"]["coordinates"] for i in imgs]
        frame = LocalFrame(*lonlat[0])
        out[seq] = view_angle(imgs, [frame.xy(*p) for p in lonlat])
    return out


def pano_views(
    metadata: dict[str, dict], records: list[dict], fov_deg: float = PANO_FOV_DEG
) -> dict[str, dict]:
    """The fetch stage's metadata with every 360 photo replaced by its right and left views
    ("<id>_right", "<id>_left"): pinhole cameras with an exact focal, the 360 photo's
    position and its rotation turned to the view. Each keeps the 360 photo's URL and a
    "pano_crop" entry ({id, side, yaw_deg, fov_deg, size}) telling texturing what to cut."""
    from scipy.spatial.transform import Rotation

    by_id = {str(r["id"]): r for r in records}
    views = sequence_view_angles(records)
    out = {}
    for img_id, meta in metadata.items():
        rec = by_id.get(str(img_id), {})
        if not (is_pano(rec) or meta.get("camera_type") in PANO_TYPES):
            out[img_id] = meta
            continue
        if meta.get("rotvec") is None:
            continue  # no orientation for the views
        view_deg = views.get(str(rec.get("sequence"))) or 0.0
        size = crop_size(meta.get("original_width") or rec.get("width"), fov_deg)
        for side, angle in PANO_SIDES.items():
            yaw = wrap_deg(angle - view_deg)
            rotvec = crop_rotvec(meta["rotvec"], yaw)
            z, y, x = Rotation.from_rotvec(rotvec).as_euler("ZYX", degrees=True)
            out[f"{img_id}_{side}"] = {
                **meta,
                "rotvec": rotvec,
                "yaw": round(float(z), 2),
                "pitch": round(float(y), 2),
                "roll": round(float(x), 2),
                "focal_length_pixels": size / 2 / math.tan(math.radians(fov_deg) / 2),
                "original_width": size,
                "original_height": size,
                "camera_type": "perspective",
                "pano_crop": {
                    "id": str(img_id),
                    "side": side,
                    "yaw_deg": round(yaw, 3),
                    "fov_deg": fov_deg,
                    "size": size,
                },
            }
    return out
