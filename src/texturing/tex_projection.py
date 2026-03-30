import json

import cv2
import numpy as np
from tqdm import tqdm
import requests
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as SciR
from trimesh import visual

from common import PipelineState
from common.MeshUtils import _get_utm_transformer

CAMERA_HEIGHT_M = 1.6
ATLAS_W = 4096
MAX_PATCH = 1024
ENABLE_OCCLUSION = True


def _build_K(focal_px: float, w: int, h: int) -> np.ndarray:
    return np.array(
        [[focal_px, 0, w / 2.0], [0, focal_px, h / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )


def _build_R(meta: dict) -> np.ndarray:
    rotvec = meta.get("rotvec")
    if rotvec is not None:
        return SciR.from_rotvec(rotvec).as_matrix()
    return SciR.from_euler(
        "ZYX", [meta["yaw"], meta["pitch"], meta["roll"]], degrees=True
    ).as_matrix()


def _project(
    Xw: np.ndarray, Cw: np.ndarray, R: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Xc = (R @ (Xw - Cw).T).T
    z = Xc[:, 2].copy()
    z[z < 1e-6] = 1e-6
    uv = (K @ np.vstack([Xc[:, 0] / z, Xc[:, 1] / z, np.ones(len(z))])).T
    return uv[:, :2], z


def _download_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from {url}")
    return img


def apply_textures(
    mesh: trimesh.Trimesh, mapillary_data: dict, bbox
) -> trimesh.Trimesh:
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    transformer = _get_utm_transformer(center_lon, center_lat)

    candidates = [
        (k, v)
        for k, v in mapillary_data.items()
        if v.get("image_url")
        and (v.get("rotvec") is not None or v.get("yaw") is not None)
        and v.get("focal_length_pixels")
        and v.get("original_width")
        and v.get("original_height")
    ]
    if not candidates:
        print("  No images with valid metadata, skipping texturing.")
        return mesh

    cam_data: list[dict] = []
    for img_id, meta in tqdm(candidates, desc="Downloading images"):
        try:
            img = _download_image(meta["image_url"])
            # img = remove_obstructions(img)
        except Exception as exc:
            tqdm.write(f"  Download failed [{img_id}]: {exc}")
            continue
        w = meta["original_width"]
        h = meta["original_height"]
        x, y = transformer.transform(meta["longitude"], meta["latitude"])
        cam_data.append(
            {
                "id": img_id,
                "img": img,
                "w": w,
                "h": h,
                "K": _build_K(meta["focal_length_pixels"], w, h),
                "R": _build_R(meta),
                "Cw": np.array([x, y, CAMERA_HEIGHT_M]),
            }
        )

    if not cam_data:
        print("  All image downloads failed, skipping texturing.")
        return mesh

    verts = mesh.vertices
    faces = mesh.faces
    _ = mesh.face_normals
    face_normals = mesh.face_normals

    verts_proj = verts.copy()
    verts_proj[:, 2] = -verts_proj[:, 2]

    occ_mesh = trimesh.Trimesh(vertices=verts_proj, faces=faces, process=False)

    best_hit: dict[int, tuple] = {}
    for ci, cd in enumerate(tqdm(cam_data, desc="Visibility")):
        Cw, R, K = cd["Cw"], cd["R"], cd["K"]
        w, h = cd["w"], cd["h"]

        for fi, face in enumerate(faces):
            V0p, V1p, V2p = verts_proj[face]
            center_p = (V0p + V1p + V2p) / 3.0

            vd = center_p - Cw
            vd_len = np.linalg.norm(vd)
            if vd_len < 1e-9:
                continue
            vd = vd / vd_len

            frontality = -np.dot(face_normals[fi], vd)
            if frontality <= 0:
                continue

            uv_img, depths = _project(np.array([V0p, V1p, V2p]), Cw, R, K)
            if not np.all(depths > 0.1):
                continue

            pts = uv_img.astype(np.int32)
            if not (
                pts[:, 0].min() >= 0
                and pts[:, 0].max() < w
                and pts[:, 1].min() >= 0
                and pts[:, 1].max() < h
            ):
                continue

            if ENABLE_OCCLUSION:
                ray_len = np.linalg.norm(center_p - Cw)
                ray_dir = (center_p - Cw) / ray_len
                locs, _, idx_tri = occ_mesh.ray.intersects_location([Cw], [ray_dir])
                if len(locs) > 0:
                    dists = np.linalg.norm(locs - Cw, axis=1)
                    nearest = np.argmin(dists)
                    if idx_tri[nearest] != fi and abs(dists[nearest] - ray_len) >= 0.5:
                        continue

            if fi not in best_hit or frontality > best_hit[fi][3]:
                best_hit[fi] = (ci, uv_img.astype(np.float32), depths, frontality)

    print(f"  {len(best_hit)} / {len(faces)} faces assigned a camera")

    patch_records: list[dict] = []
    for fi, (ci, uv_img, depths, _) in best_hit.items():
        face = faces[fi]
        V0p, V1p, V2p = verts_proj[face]
        cd = cam_data[ci]
        Cw, R, K = cd["Cw"], cd["R"], cd["K"]

        e1 = V1p - V0p
        e1_len = np.linalg.norm(e1)
        if e1_len < 1e-6:
            continue
        e1_hat = e1 / e1_len

        n_cross = np.cross(V1p - V0p, V2p - V0p)
        n_len = np.linalg.norm(n_cross)
        if n_len < 1e-6:
            continue
        n_hat = n_cross / n_len
        e2_hat = np.cross(e1_hat, n_hat)
        if e2_hat[2] < 0:
            e2_hat = -e2_hat

        v1_s = np.dot(V1p - V0p, e1_hat)
        v1_t = np.dot(V1p - V0p, e2_hat)
        v2_s = np.dot(V2p - V0p, e1_hat)
        v2_t = np.dot(V2p - V0p, e2_hat)

        s_all = [0.0, v1_s, v2_s]
        t_all = [0.0, v1_t, v2_t]
        s_min, s_max = min(s_all), max(s_all)
        t_min, t_max = min(t_all), max(t_all)
        if s_max <= s_min or t_max <= t_min:
            continue

        face_w_m = s_max - s_min
        face_h_m = t_max - t_min
        img_bw = max(uv_img[:, 0].max() - uv_img[:, 0].min(), 1.0)
        img_bh = max(uv_img[:, 1].max() - uv_img[:, 1].min(), 1.0)
        px_per_m = min(img_bw / face_w_m, img_bh / face_h_m)
        patch_w = int(np.clip(face_w_m * px_per_m, 2, MAX_PATCH))
        patch_h = int(np.clip(face_h_m * px_per_m, 2, MAX_PATCH))

        M = np.column_stack([R @ e1_hat, R @ e2_hat, R @ (V0p - Cw)])
        H = K @ M

        ds = (s_max - s_min) / patch_w
        dt = (t_max - t_min) / patch_h
        S = np.array([[ds, 0, s_min], [0, -dt, t_max], [0, 0, 1]], dtype=np.float64)
        H_atlas = H @ S

        patch_bgr = cv2.warpPerspective(
            cd["img"],
            H_atlas,
            (patch_w, patch_h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

        def to_px(s, t, _ds=ds, _dt=dt, _s_min=s_min, _t_max=t_max):
            return (s - _s_min) / _ds, (_t_max - t) / _dt

        patch_records.append(
            {
                "fi": fi,
                "patch": patch_rgb,
                "pw": patch_w,
                "ph": patch_h,
                "v0_px": to_px(0.0, 0.0),
                "v1_px": to_px(v1_s, v1_t),
                "v2_px": to_px(v2_s, v2_t),
            }
        )

    cx, cy, row_h = 0, 0, 0
    for p in patch_records:
        if cx + p["pw"] > ATLAS_W:
            cx = 0
            cy += row_h
            row_h = 0
        p["ax"] = cx
        p["ay"] = cy
        cx += p["pw"]
        row_h = max(row_h, p["ph"])
    photo_h = cy + row_h

    GRAY_H = 16
    atlas_h = photo_h + GRAY_H
    atlas = np.full((atlas_h, ATLAS_W, 3), 180, dtype=np.uint8)
    for p in patch_records:
        atlas[p["ay"] : p["ay"] + p["ph"], p["ax"] : p["ax"] + p["pw"]] = p["patch"]

    DEFAULT_U = 0.5
    DEFAULT_V = 1.0 - (photo_h + GRAY_H / 2) / atlas_h

    n_all = len(faces)
    new_verts = np.zeros((n_all * 3, 3))
    new_faces = np.arange(n_all * 3, dtype=np.int64).reshape(n_all, 3)
    new_uvs = np.full((n_all * 3, 2), [DEFAULT_U, DEFAULT_V])

    for i, face in enumerate(faces):
        new_verts[i * 3 : i * 3 + 3] = verts[face]

    for p in patch_records:
        fi = p["fi"]
        for j, key in enumerate(("v0_px", "v1_px", "v2_px")):
            ax_v, ay_v = p[key]
            u = (p["ax"] + ax_v) / ATLAS_W
            v = 1.0 - (p["ay"] + ay_v) / atlas_h
            new_uvs[fi * 3 + j] = [np.clip(u, 0, 1), np.clip(v, 0, 1)]

    tex_pil = Image.fromarray(atlas)
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    mat = visual.material.PBRMaterial(baseColorTexture=tex_pil, doubleSided=True)
    new_mesh.visual = visual.TextureVisuals(uv=new_uvs, material=mat, image=tex_pil)
    print(f"  Atlas: {ATLAS_W}x{atlas_h} px, {len(patch_records)} face patches")
    return new_mesh


def tex_projection(value, state: PipelineState):
    mapillary_data = state.get_metadata("provider_data", {})["mapillary"]
    with open("test.json", "w") as f:
        json.dump(mapillary_data, f, indent=2)
    bbox = state.require_metadata("bbox")
    mesh = apply_textures(value, mapillary_data, bbox)
    state.require_metadata("progress_monitor").next()
    return mesh
