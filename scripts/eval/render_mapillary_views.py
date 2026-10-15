"""Render a pipeline GLB from the cameras of Mapillary photos.

For each Mapillary image, places Blender's camera where the photo was taken
(Mapillary's computed position and rotation, the photo's focal length) and
renders the GLB at the photo's size with the settings from setup_render.py.

Writes to --out, per image id:
    <id>.jpg           the Mapillary photo
    <id>_render.png    textured model, transparent background
    <id>_grey.png      untextured grey model (a baseline)
    <id>_overlay.jpg   photo with the model outline in red, to check alignment

Usage (from the repo root):
    poetry run python scripts/eval/render_mapillary_views.py outputs/way_291762178.json ID [ID ...]
    poetry run python scripts/eval/render_mapillary_views.py outputs/way_291762178.json --rank 16

--rank N renders every photo the export downloaded at low resolution, ranks
them by how much of the frame the building fills, and prints the top N.
"""

import argparse
import json
import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from pyproj import Transformer
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender_render_views.py"
FIELDS = "id,camera_type,computed_geometry,geometry,computed_rotation,camera_parameters,width,height,thumb_original_url"
EYE_HEIGHT_M = 1.6

# OpenCV camera axes (x right, y down, z forward) -> Blender camera axes (y up, looks down -z)
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])


def fetch_metadata(image_ids: list[str]) -> dict[str, dict]:
    token = os.environ["MAPILLARY_ACCESS_TOKEN"]
    meta = {}
    for i in range(0, len(image_ids), 50):
        resp = requests.get(
            "https://graph.mapillary.com/images",
            params={"access_token": token, "image_ids": ",".join(image_ids[i : i + 50]), "fields": FIELDS},
            timeout=60,
        )
        resp.raise_for_status()
        meta.update({d["id"]: d for d in resp.json()["data"]})
    return meta


def camera_job(
    img: dict, sidecar: dict, transformer: Transformer, scale: float = 1.0, pose: dict | None = None
) -> dict | None:
    """Blender camera for a photo: Mapillary's computed pose, or `pose` (an entry of the
    sidecar's camera_poses) when given."""
    if img.get("camera_type") not in (None, "perspective") or not img.get("camera_parameters"):
        return None
    if pose is not None:
        rotvec, lon, lat = pose["rotvec"], pose["longitude"], pose["latitude"]
        height = pose.get("camera_height_m") or EYE_HEIGHT_M
        lens_mm = pose["focal_length_pixels"] / max(pose["original_width"], pose["original_height"]) * 36.0
    else:
        rotvec = img.get("computed_rotation")
        geom = img.get("computed_geometry") or img.get("geometry")
        if rotvec is None or geom is None:
            return None
        lon, lat = geom["coordinates"]
        height, lens_mm = EYE_HEIGHT_M, img["camera_parameters"][0] * 36.0
    easting, northing = transformer.transform(lon, lat)

    # Mapillary rotations are relative to true north; the model uses UTM grid north.
    north_e, north_n = transformer.transform(lon, lat + 1e-3)
    convergence = np.arctan2(north_e - easting, north_n - northing)
    true_to_grid = Rotation.from_rotvec([0, 0, -convergence]).as_matrix()

    world_to_cam = Rotation.from_rotvec(rotvec).as_matrix()
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = true_to_grid @ world_to_cam.T @ CV_TO_BLENDER
    glb_to_utm = np.array(sidecar["glb_to_utm"])
    cam_to_world[:3, 3] = [
        easting - glb_to_utm[0, 3],
        northing - glb_to_utm[1, 3],
        -glb_to_utm[2, 3] + height,
    ]
    w, h = img["width"], img["height"]
    return {
        "width": max(1, round(w * scale)),
        "height": max(1, round(h * scale)),
        "matrix": cam_to_world.tolist(),
        "lens_mm": lens_mm,
    }


def run_blender(glb: Path, jobs: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(jobs, f)
    subprocess.run(
        [BLENDER, "-b", "--python", str(BLENDER_SCRIPT), "--", str(glb), f.name],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    os.unlink(f.name)


def make_overlay(photo: Image.Image, render_path: Path, out_path: Path) -> float:
    rgba = np.array(Image.open(render_path))
    mask = rgba[..., 3] > 127
    out = np.array(photo.convert("RGB"))
    thickness = max(3, photo.width // 400)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((thickness, thickness), np.uint8)) > 0
    out[edge] = [255, 0, 0]
    preview = Image.fromarray(out)
    preview.thumbnail((1200, 1200))
    preview.save(out_path)
    return float(mask.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sidecar", type=Path, help="JSON written by export_single_building.py")
    parser.add_argument("image_ids", nargs="*", help="Mapillary image ids")
    parser.add_argument("--rank", type=int, default=0, help="Rank downloaded photos by coverage, show top N")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--own-poses",
        action="store_true",
        help="Place the camera at the pose the export textured with (the sidecar's camera_poses)",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    sidecar = json.loads(args.sidecar.read_text())
    glb = args.sidecar.with_suffix(".glb")
    out_dir = args.out or args.sidecar.parent / f"{args.sidecar.stem}_views"
    out_dir.mkdir(parents=True, exist_ok=True)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{sidecar['utm_epsg']}", always_xy=True)
    used = sidecar.get("texture_source_ids", {})

    if args.rank:
        meta = fetch_metadata(sidecar["mapillary_fetched_ids"])
        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            for img_id, img in meta.items():
                job = camera_job(img, sidecar, transformer, scale=256 / img["width"])
                if job:
                    jobs.append({**job, "out": f"{tmp}/{img_id}.png"})
            run_blender(glb, jobs)
            coverage = {
                Path(j["out"]).stem: float((np.array(Image.open(j["out"]))[..., 3] > 127).mean()) for j in jobs
            }
        print(f"{'coverage':>8}  {'image id':<20} texture faces")
        for img_id, cov in sorted(coverage.items(), key=lambda kv: -kv[1])[: args.rank]:
            print(f"{cov * 100:7.1f}%  {img_id:<20} {used.get(img_id, 0)}")
        return

    meta = fetch_metadata(args.image_ids)
    jobs, photos = [], {}
    poses = sidecar.get("camera_poses", {}) if args.own_poses else {}
    for img_id in args.image_ids:
        if args.own_poses and img_id not in poses:
            print(f"Skipping {img_id}: no pose in the sidecar's camera_poses")
            continue
        job = camera_job(meta[img_id], sidecar, transformer, pose=poses.get(img_id))
        if job is None:
            print(f"Skipping {img_id}: not a regular photo or missing camera data")
            continue
        photo_bytes = requests.get(meta[img_id]["thumb_original_url"], timeout=60).content
        (out_dir / f"{img_id}.jpg").write_bytes(photo_bytes)
        photos[img_id] = Image.open(BytesIO(photo_bytes))
        jobs.append({**job, "out": str(out_dir / f"{img_id}_render.png")})
        jobs.append({**job, "out": str(out_dir / f"{img_id}_grey.png"), "grey": True})
    run_blender(glb, jobs)

    for img_id, photo in photos.items():
        cov = make_overlay(photo, out_dir / f"{img_id}_render.png", out_dir / f"{img_id}_overlay.jpg")
        print(f"{img_id}: building covers {cov * 100:.1f}% of the frame, texture faces from it: {used.get(img_id, 0)}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
