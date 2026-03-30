from dotenv import load_dotenv
from common import ProgressMonitor, PipelineChain, export_to_glb
from common.profiler import PipelineProfiler
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES
from common.MeshUtils import _get_utm_transformer

from mesh_builder.extrude import extrude_buildings
from data_ingest.ingest import ingest_data
from mesh_builder.extrude import build_mesh
from texturing.tex_projection import tex_projection
from tkinter import Tk, filedialog

import argparse
import os
import numpy as np
import trimesh

_env = os.environ.get("APP_ENV", "").lower()
_env_file = {
    "development": ".env.development",
    "production": ".env.production",
}.get(_env, ".env")
load_dotenv(_env_file)


def progress():
    bbox = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)

    # Register a blue print of each task
    progress = ProgressMonitor()
    progress.add_task("Retrieve data from OSM and Mapillary")
    progress.add_task("Transform OSM to Mesh data")
    progress.add_task("Texture mesh")
    progress.add_task("Export mesh file")
    progress.add_task("FINISHED!")

    # Pipeline
    cl = OSMClient()

    data = cl.fetch(bbox, [OSM_MAP_FEATURES.BUILDING])
    progress.next()

    extrude_buildings(data, bbox)
    progress.next()
    progress.next()

    progress.next()


# MAIN PIPELINE RUN
pipeline_progress = ProgressMonitor()
pipeline_progress.add_task("Fetching data from OSM and Mapillary...")
pipeline_progress.add_task("Retrieved data from OSM")
pipeline_progress.add_task("Transform OSM to Mesh data")
pipeline_progress.add_task("Texture Mesh")
pipeline_progress.add_task("Export mesh file")


def ask_save_path(default_name: str = "combined.glb") -> str | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.asksaveasfilename(
        title="Choose where to save",
        defaultextension=".glb",
        initialfile=default_name,
        filetypes=[("GLB files", "*.glb"), ("All files", "*.*")],
    )

    root.destroy()
    return path or None


def export_mesh(value, state):
    bbox = state.require_metadata("bbox")
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    transformer = _get_utm_transformer(center_lon, center_lat)
    ox, oy = transformer.transform(center_lon, center_lat)

    value.vertices -= np.array([ox, oy, 0.0])
    value.vertices[:, 2] *= -1
    center = value.bounding_box.centroid
    value.apply_translation(-center)
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.radians(-90.0),
        direction=[1.0, 0.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )
    value.apply_transform(rotation)

    path = ask_save_path()
    if path is not None:
        export_to_glb(value, path or "combined.glb")

    state.require_metadata("progress_monitor").next()


run_pipeline = PipelineChain()
run_pipeline.add_stage("fetch", ingest_data)
run_pipeline.add_stage("build_mesh", build_mesh)
run_pipeline.add_stage("texturing", tex_projection)
run_pipeline.add_stage("export", export_mesh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", metavar="FILENAME", default=None)
    parser.add_argument("--no-seg", action="store_true", default=False)
    args = parser.parse_args()

    min_lon = float(input("WEST (Minimum Longitude): "))
    min_lat = float(input("SOUTH (Minimum Latitude): "))
    max_lon = float(input("EAST (Minimum Longitude): "))
    max_lat = float(input("NORTH (Minimum Latitude): "))

    bbox = BoundingBox(min_lat, max_lat, min_lon, max_lon)

    profiler = (
        PipelineProfiler(pipeline_name="photorealistic-street-maps")
        if args.profile
        else None
    )
    run_pipeline.run(
        bbox,
        metadata={
            "bbox": bbox,
            "progress_monitor": pipeline_progress,
            "remove_obstructions": not args.no_seg,
        },
        profiler=profiler,
    )
    if profiler is not None:
        profiler.save(args.profile)


if __name__ == "__main__":
    main()
