from common import ProgressMonitor
from common.profiler import PipelineProfiler
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES
from common.env import load_app_env

from mesh_builder.extrude import extrude_buildings
from pipeline import build_pipeline

import argparse

load_app_env()


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
    # Imported lazily so headless runs (e.g. Docker with --output) never need Tk.
    from tkinter import Tk, filedialog

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


run_pipeline = build_pipeline()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", metavar="FILENAME", default=None)
    parser.add_argument("--no-seg", action="store_true", default=False)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help="Bounding box in degrees. Prompts interactively if omitted.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Where to write the .glb. Opens a save dialog if omitted.",
    )
    parser.add_argument(
        "--dump-mapillary",
        metavar="PATH",
        default=None,
        help="Write the raw Mapillary metadata to this JSON file for debugging.",
    )
    args = parser.parse_args()

    if args.bbox is not None:
        min_lon, min_lat, max_lon, max_lat = args.bbox
    else:
        min_lon = float(input("WEST (Minimum Longitude): "))
        min_lat = float(input("SOUTH (Minimum Latitude): "))
        max_lon = float(input("EAST (Maximum Longitude): "))
        max_lat = float(input("NORTH (Maximum Latitude): "))

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
            "output_path": args.output,
            "ask_output_path": ask_save_path,
            "mapillary_dump_path": args.dump_mapillary,
        },
        profiler=profiler,
    )
    if profiler is not None:
        profiler.save(args.profile)


if __name__ == "__main__":
    main()
