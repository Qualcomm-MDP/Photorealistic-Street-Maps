"""Run the pipeline on a single OSM building and export it as a GLB.

Writes <out>.glb plus <out>.json containing the building's OSM metadata and
`glb_to_utm`, a 4x4 matrix mapping GLB vertex coordinates to UTM
(easting, northing, up) in metres, for aligning against reference models.

With --poses it also records the camera pose of every image (`camera_poses`), so
scripts/eval/render_mapillary_views.py --own-poses renders from the poses texturing used.
--exclude-images keeps held-out photos out of texturing while still recording their pose.

Usage (from the repo root):
    poetry run python scripts/export_single_building.py 291762178 --no-seg
    poetry run python scripts/export_single_building.py 467844847 --no-seg --poses colmap \\
        --exclude-images ID1,ID2 --out outputs/ab/colmap/way_467844847.glb
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

import main as pipeline_main  # noqa: E402
from common import PipelineChain  # noqa: E402
from common.constants import OVERPASS_URL, USER_AGENT, BoundingBox  # noqa: E402
from common.MeshUtils import _get_utm_transformer  # noqa: E402
from common.providers import MapillaryClient  # noqa: E402
from mesh_builder.extrude import build_mesh  # noqa: E402
from pose_alignment.align_poses import POSE_SOURCES, align_poses  # noqa: E402
from texturing.tex_projection import tex_projection  # noqa: E402

MAPILLARY_FIELDS = [
    "id",
    "thumb_original_url",
    "sequence_id",
    "geometry",
    "computed_geometry",
    "computed_compass_angle",
    "computed_rotation",
    "camera_parameters",
    "width",
    "height",
]


def fetch_building(osm_id: int, retries: int = 5) -> dict:
    query = f"[out:json][timeout:25];way({osm_id});out body geom;"
    for attempt in range(retries):
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        if resp.status_code in (429, 504) and attempt < retries - 1:
            print(f"Overpass returned {resp.status_code}, retrying in 15s …")
            time.sleep(15)
            continue
        resp.raise_for_status()
        break

    elements = [e for e in resp.json()["elements"] if "geometry" in e]
    if not elements:
        raise ValueError(f"OSM way {osm_id} not found or has no geometry")
    return elements[0]


def expand_bbox(building: dict, margin_m: float) -> BoundingBox:
    lats = [p["lat"] for p in building["geometry"]]
    lons = [p["lon"] for p in building["geometry"]]
    dlat = margin_m / 111_320
    dlon = margin_m / (111_320 * math.cos(math.radians(sum(lats) / len(lats))))
    return BoundingBox(
        min_lat=min(lats) - dlat,
        max_lat=max(lats) + dlat,
        min_lon=min(lons) - dlon,
        max_lon=max(lons) + dlon,
    )


def make_fetch_stage(building: dict, limit: int):
    def fetch(bbox: BoundingBox, state):
        progress = state.require_metadata("progress_monitor")
        progress.next()
        mapillary_data = MapillaryClient().fetch(bbox, MAPILLARY_FIELDS, limit=limit)
        progress.next()
        print(f"  {len(mapillary_data)} Mapillary images within the bounding box")
        data = {"osm": {"elements": [building]}, "mapillary": mapillary_data}
        state.set_metadata("provider_data", data)
        state.set_metadata("mapillary_fetched_ids", sorted(mapillary_data))
        return data

    return fetch


def make_export_stage(out_path: Path):
    def export(mesh, state):
        utm = mesh.vertices.copy()
        utm[:, 2] *= -1
        pipeline_main.ask_save_path = lambda *a, **k: str(out_path)
        pipeline_main.export_mesh(mesh, state)

        glb = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
        solution, *_ = np.linalg.lstsq(glb, utm, rcond=None)
        glb_to_utm = np.eye(4)
        glb_to_utm[:3, :] = solution.T
        state.set_metadata("glb_to_utm", glb_to_utm)
        return mesh

    return export


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("osm_id", type=int, help="OSM way id of the building")
    parser.add_argument("--out", type=Path, default=None, help="Output .glb path")
    parser.add_argument(
        "--margin",
        type=float,
        default=50.0,
        help="Metres around the footprint to search for Mapillary photos",
    )
    parser.add_argument("--limit", type=int, default=200, help="Max Mapillary photos")
    parser.add_argument("--no-seg", action="store_true", default=False)
    parser.add_argument("--poses", choices=POSE_SOURCES, default="raw", help="Camera poses for texturing")
    parser.add_argument("--aligned-only", action="store_true", help="With --poses colmap, only aligned images")
    parser.add_argument("--alignment-dir", metavar="DIR", default=None, help="With --poses colmap, reuse this ROI")
    parser.add_argument("--exclude-images", default="", help="Comma-separated image ids kept out of texturing")
    parser.add_argument("--height", type=float, default=None, help="Building height in metres (overrides OSM)")
    args = parser.parse_args()
    exclude = [i for i in args.exclude_images.split(",") if i]

    building = fetch_building(args.osm_id)
    if args.height is not None:
        building.setdefault("tags", {})["height"] = str(args.height)
    name = building.get("tags", {}).get("name", f"way_{args.osm_id}")
    print(f"Building: {name} (way {args.osm_id})")

    out_path = args.out or REPO_ROOT / "outputs" / f"way_{args.osm_id}.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bbox = expand_bbox(building, args.margin)
    pipeline = PipelineChain("single-building")
    pipeline.add_stage("fetch", make_fetch_stage(building, args.limit))
    pipeline.add_stage("align_poses", align_poses)
    pipeline.add_stage("build_mesh", build_mesh)
    pipeline.add_stage("texturing", tex_projection)
    pipeline.add_stage("export", make_export_stage(out_path))

    state = pipeline.run(
        bbox,
        metadata={
            "bbox": bbox,
            "progress_monitor": pipeline_main.pipeline_progress,
            "remove_obstructions": not args.no_seg,
            "pose_source": args.poses,
            "aligned_only": args.aligned_only,
            "alignment_dir": args.alignment_dir,
            "exclude_images": exclude,
        },
    )

    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    utm_epsg = _get_utm_transformer(center_lon, center_lat).target_crs.to_epsg()
    footprint = [[p["lat"], p["lon"]] for p in building["geometry"]]
    sidecar = {
        "osm_id": args.osm_id,
        "osm_type": "way",
        "name": name,
        "tags": building.get("tags", {}),
        "footprint_latlon": footprint,
        "centroid_latlon": np.mean(footprint, axis=0).tolist(),
        "utm_epsg": utm_epsg,
        "glb_to_utm": state.require_metadata("glb_to_utm").tolist(),
        "glb_axes": "x=east, y=up, z=south (glTF Y-up)",
        "search_bbox": vars(bbox),
        "mapillary_images": len(state.require_metadata("mapillary_fetched_ids")),
        "obstruction_removal": not args.no_seg,
        "texture_source_ids": state.get_metadata("texture_sources", {}),
        "mapillary_fetched_ids": state.require_metadata("mapillary_fetched_ids"),
        "pose_source": args.poses,
        "pose_alignment": state.get_metadata("pose_alignment", None),
        "excluded_images": exclude,
        "camera_poses": state.get_metadata("camera_poses", {}),
    }
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path} and {sidecar_path}")


if __name__ == "__main__":
    main()
