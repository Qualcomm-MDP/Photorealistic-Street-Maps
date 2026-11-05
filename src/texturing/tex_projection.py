import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as SciR
from trimesh import visual

from common import PipelineState
from common.MeshUtils import _get_utm_transformer
from common.reporter import Reporter, get_reporter
from naip.naip import roof_faces, roof_patches
from data_ingest.pano import PANO_TYPES, crop_view
from texturing.blend import bake_walls
from segmentation import build_obstruction_mask, remove_obstructions
from segmentation.model import LabelCache, model_id_or_default

CAMERA_HEIGHT_M = 1.6
ATLAS_W = 4096
MAX_PATCH = 1024
ENABLE_OCCLUSION = True
# Colour (0-255) of faces no street photo covers: roofs, back walls, gaps.
UNTEXTURED_GREY = 225
# With fill_obstructions off, each photo's obstruction mask is kept at 1/MASK_SCALE size to
# score how much of a face it hides; a face then takes the photo with the best
# frontality x unobstructed share.
MASK_SCALE = 8


def camera_position(
    x: float, y: float, terrain=None, height_m: float | None = None
) -> np.ndarray:
    """Camera centre in projection space: height_m (a measured camera height, e.g. the
    pose alignment's; default CAMERA_HEIGHT_M) above the ground (the terrain's, or 0)."""
    ground_z = float(terrain.ground(x, y)) if terrain is not None else 0.0
    return np.array(
        [x, y, ground_z + (CAMERA_HEIGHT_M if height_m is None else height_m)]
    )


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


def _download_image(url: str) -> tuple[np.ndarray, bytes]:
    """The decoded photo and the file exactly as downloaded (or read from disk)."""
    if not url.startswith(("http://", "https://")):
        img = cv2.imread(url, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image {url}")
        return img, Path(url).read_bytes()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from {url}")
    return img, resp.content


# Barycentric sample points inside a triangle (a 6-step grid, 28 points).
_BARY = np.array(
    [(i / 6, j / 6, 1 - (i + j) / 6) for i in range(7) for j in range(7 - i)]
)


def _obstructed_share(mask: np.ndarray, uv: np.ndarray, width: int) -> float:
    """Share of a face's projected triangle (uv, 3 x 2 image pixels) that a photo's
    obstruction mask (scaled to mask.shape from an image `width` px wide) covers."""
    k = mask.shape[1] / width
    pts = (_BARY @ uv) * k
    x = np.clip(pts[:, 0].astype(int), 0, mask.shape[1] - 1)
    y = np.clip(pts[:, 1].astype(int), 0, mask.shape[0] - 1)
    return float(mask[y, x].mean()) / 255.0


def _save_preview(img: np.ndarray, path: Path, max_width: int = 640) -> None:
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def apply_textures(
    mesh: trimesh.Trimesh,
    mapillary_data: dict,
    bbox,
    remove_obs: bool = True,
    seg_model: str | None = None,
    seg_device: str | None = None,
    image_dir: str | Path | None = None,
    max_saved_images: int = 12,
    reporter: Reporter | None = None,
    terrain=None,
    aerial=None,
    image_store=None,
    seg_labels: LabelCache | None = None,
    fill_obstructions: bool = True,
    texture_mode: str = "blend",
) -> trimesh.Trimesh:
    """Project street photos onto the mesh.

    texture_mode "blend" (default) bakes every wall from all the photos that see it
    (texturing.blend: per-texel weighted average, seamless across a wall's triangles);
    "best" warps each face from its single most square-on photo (the earlier behaviour).

    If image_dir is set, the first max_saved_images photos are also saved there
    as small JPEG previews (original and, when obstruction removal runs, cleaned),
    with an index in images.json.

    If terrain is given, each camera sits CAMERA_HEIGHT_M above the ground at
    its position instead of above z = 0.

    If aerial (a naip.Ortho) is given, roofs are textured from it instead of
    from street photos.

    image_store (data_ingest.image_store.ImageStore) and seg_labels hold photos and
    class maps the pose alignment already made: those photos are not downloaded or
    segmented again.

    With remove_obs, the obstructions (cars, people, poles, ...) are segmented. With
    fill_obstructions they are painted over (LaMa); without it the photos stay as taken
    and the masks steer which photo textures a face: the most square-on view that the
    obstructions hide least.
    """
    # Full-resolution photos are ~12 MP (~36 MB decoded) and an area can have
    # ~1000 of them, so they're kept on disk and loaded one at a time.
    with tempfile.TemporaryDirectory(prefix="psm-textures-") as work_dir:
        return _apply_textures(
            mesh,
            mapillary_data,
            bbox,
            remove_obs,
            seg_model,
            seg_device,
            image_dir,
            max_saved_images,
            Path(work_dir),
            reporter or get_reporter(),
            terrain,
            aerial,
            image_store,
            seg_labels,
            fill_obstructions,
            texture_mode,
        )


def _apply_textures(
    mesh: trimesh.Trimesh,
    mapillary_data: dict,
    bbox,
    remove_obs: bool,
    seg_model: str | None,
    seg_device: str | None,
    image_dir: str | Path | None,
    max_saved_images: int,
    work_dir: Path,
    reporter: Reporter,
    terrain=None,
    aerial=None,
    image_store=None,
    seg_labels: LabelCache | None = None,
    fill_obstructions: bool = True,
    texture_mode: str = "blend",
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
        # a 360 photo is used through its right and left views (data_ingest.pano)
        and v.get("camera_type") not in PANO_TYPES
    ]
    if not candidates and aerial is None:
        reporter.info("  No images with valid metadata, skipping texturing.")
        return mesh

    saved: list[dict] = []
    if image_dir is not None:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

    # class maps from the alignment, when they come from the model obstruction removal uses
    cached_labels = None
    if remove_obs and seg_labels is not None:
        cached_model, names = seg_labels.model()
        if cached_model == model_id_or_default(seg_model):
            cached_labels = names
    reused = {"photos": 0, "class maps": 0}

    cam_data: list[dict] = []
    pano: tuple[str, np.ndarray] | None = None  # the last 360 photo: (id, image)
    for img_id, meta in reporter.progress(candidates, desc="Downloading images"):
        crop = meta.get("pano_crop")  # a view of a 360 photo (data_ingest.pano)
        source_id = crop["id"] if crop else str(img_id)
        try:
            if crop and pano is not None and pano[0] == source_id:
                img = pano[1]  # the other view of the 360 photo just downloaded
            else:
                # the stored original stands in for its URL (an image the alignment
                # added already points at its own frame, whose size its pose was
                # solved for)
                stored = (
                    image_store is not None
                    and str(meta["image_url"]).startswith(("http://", "https://"))
                    and image_store.has(source_id)
                )
                if stored:
                    img, raw = _download_image(str(image_store.path(source_id)))
                    reused["photos"] += 1
                else:
                    img, raw = _download_image(meta["image_url"])
            if crop:
                pano = (source_id, img)
                img = crop_view(img, crop["yaw_deg"], crop["fov_deg"], crop["size"])
                raw = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[
                    1
                ].tobytes()
        except Exception as exc:
            reporter.info(f"  Download failed [{img_id}]: {exc}")
            continue
        # Folder for this photo's previews, or None once enough are saved.
        preview_dir = image_dir if len(saved) < max_saved_images else None
        if preview_dir is not None:
            _save_preview(img, preview_dir / f"{img_id}_original.jpg")
        if remove_obs:
            labels = None
            if cached_labels is not None and seg_labels is not None:
                seg = seg_labels.get(str(img_id), img.shape[:2])
                if seg is not None:
                    labels = (seg, cached_labels)
                    reused["class maps"] += 1
            if fill_obstructions:
                img = remove_obstructions(img, seg_model, seg_device, reporter, labels)
            else:
                mask = build_obstruction_mask(
                    img, seg_model, seg_device, reporter, labels
                )
                small_mask = cv2.resize(
                    mask,
                    (
                        max(1, mask.shape[1] // MASK_SCALE),
                        max(1, mask.shape[0] // MASK_SCALE),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                if preview_dir is not None:
                    shaded = img.copy()
                    shaded[mask > 0] = (0.4 * shaded[mask > 0]).astype(np.uint8)
                    img_for_preview = shaded
            if preview_dir is not None:
                _save_preview(
                    img if fill_obstructions else img_for_preview,
                    preview_dir / f"{img_id}_cleaned.jpg",
                )
        if preview_dir is not None:
            yaw = meta.get("yaw")
            saved.append(
                {
                    "id": str(img_id),
                    "heading": None if yaw is None else round(float(yaw) % 360),
                    "original": f"{img_id}_original.jpg",
                    "cleaned": f"{img_id}_cleaned.jpg" if remove_obs else None,
                }
            )
        # Spool to disk without losing quality: the downloaded file as is, or a
        # lossless PNG once obstruction removal has changed the pixels.
        if remove_obs and fill_obstructions:
            img_path = work_dir / f"{img_id}.png"
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            img_path = work_dir / f"{img_id}.jpg"
            img_path.write_bytes(raw)
        del img, raw
        w = meta["original_width"]
        h = meta["original_height"]
        x, y = transformer.transform(meta["longitude"], meta["latitude"])
        cam_data.append(
            {
                "id": img_id,
                "img_path": img_path,
                "w": w,
                "h": h,
                "K": _build_K(meta["focal_length_pixels"], w, h),
                "R": _build_R(meta),
                "mask": small_mask if remove_obs and not fill_obstructions else None,
                # an aligned camera's height is above the road under it, so on terrain
                # it goes on top of the ground there, like the default height
                "Cw": camera_position(x, y, terrain, meta.get("camera_height_m")),
            }
        )

    if image_dir is not None:
        with open(image_dir / "images.json", "w") as f:
            json.dump(saved, f, indent=2)
    if any(reused.values()):
        reporter.info(
            "  Reused from the pose alignment: "
            + ", ".join(f"{n} {k}" for k, n in reused.items())
        )

    if not cam_data and aerial is None:
        reporter.info("  All image downloads failed, skipping texturing.")
        return mesh

    verts = mesh.vertices
    faces = mesh.faces
    _ = mesh.face_normals
    face_normals = mesh.face_normals

    verts_proj = verts.copy()
    verts_proj[:, 2] = -verts_proj[:, 2]

    occ_mesh = trimesh.Trimesh(vertices=verts_proj, faces=faces, process=False)

    roof_idx = roof_faces(mesh) if aerial is not None else np.array([], dtype=int)
    skip = set(roof_idx.tolist())

    patch_records: list[dict]
    if texture_mode == "blend":
        patch_records, stats = bake_walls(
            verts_proj, faces, face_normals, cam_data, skip, reporter, UNTEXTURED_GREY
        )
        reporter.info(
            f"  {stats.get('textured_walls', 0)} / {stats.get('walls', 0)} wall faces blended from "
            f"the photos ({100 * stats.get('coverage', 0):.0f}% of their texels seen)"
        )
    else:
        best_hit: dict[int, tuple] = {}
        for ci, cd in enumerate(reporter.progress(cam_data, desc="Visibility")):
            Cw, R, K = cd["Cw"], cd["R"], cd["K"]
            w, h = cd["w"], cd["h"]

            for fi, face in enumerate(faces):
                if fi in skip:
                    continue
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
                        if (
                            idx_tri[nearest] != fi
                            and abs(dists[nearest] - ray_len) >= 0.5
                        ):
                            continue

                score = frontality
                if cd["mask"] is not None:
                    score *= 1.0 - _obstructed_share(cd["mask"], uv_img, w)
                if fi not in best_hit or score > best_hit[fi][3]:
                    best_hit[fi] = (ci, uv_img.astype(np.float32), depths, score)

        reporter.info(f"  {len(best_hit)} / {len(faces)} faces assigned a camera")

        patch_records = []
        # Group faces by camera so each photo is read from disk once.
        loaded_ci, loaded_img = None, None
        for fi, (ci, uv_img, depths, _) in sorted(
            best_hit.items(), key=lambda item: item[1][0]
        ):
            if ci != loaded_ci:
                loaded_ci, loaded_img = ci, cv2.imread(str(cam_data[ci]["img_path"]))
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
                loaded_img,
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

    if aerial is not None:
        roofs = roof_patches(verts_proj, faces, roof_idx, aerial)
        reporter.info(f"  {len(roofs)} / {len(roof_idx)} roof faces from NAIP")
        patch_records.extend(roofs)

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
    atlas = np.full((atlas_h, ATLAS_W, 3), UNTEXTURED_GREY, dtype=np.uint8)
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
    reporter.info(f"  Atlas: {ATLAS_W}x{atlas_h} px, {len(patch_records)} face patches")
    return new_mesh


def tex_projection(value, state: PipelineState):
    mapillary_data = state.get_metadata("provider_data", {})["mapillary"]
    dump_path = state.get_metadata("mapillary_dump_path", None)
    if dump_path:
        with open(dump_path, "w") as f:
            json.dump(mapillary_data, f, indent=2)
    bbox = state.require_metadata("bbox")
    mesh = apply_textures(
        value,
        mapillary_data,
        bbox,
        remove_obs=state.get_metadata("remove_obstructions", True),
        fill_obstructions=state.get_metadata("fill_obstructions", True),
        texture_mode=state.get_metadata("texture_mode", "blend"),
        seg_model=state.get_metadata("seg_model", None),
        seg_device=state.get_metadata("seg_device", None),
        terrain=state.get_metadata("terrain", None),
        aerial=state.get_metadata("naip", None),
        image_dir=state.get_metadata("image_dir", None),
        reporter=get_reporter(state),
        image_store=state.get_metadata("image_store", None),
        seg_labels=state.get_metadata("segmentation_labels", None),
    )
    state.require_metadata("progress_monitor").next()
    return mesh
