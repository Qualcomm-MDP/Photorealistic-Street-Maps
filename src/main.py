from common import ProgressMonitor
from common.profiler import PipelineProfiler
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES
from common.env import load_app_env

from mesh_builder.extrude import extrude_buildings
from pipeline import build_pipeline
from pose_alignment.align_poses import POSE_SOURCES

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
    parser.add_argument(
        "--no-seg",
        action="store_true",
        default=False,
        help="Skip obstruction removal in texturing (no segmentation, no inpainting)",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        default=False,
        help="Segment the obstructions (cars, people, poles, ...) but do not paint them over "
        "(LaMa): each face takes the photo where they hide it least",
    )
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
    parser.add_argument(
        "--terrain",
        action="store_true",
        default=False,
        help="Place buildings and cameras on real ground heights (Terrarium "
        "elevation tiles) and add the ground to the .glb. Off by default.",
    )
    parser.add_argument(
        "--naip",
        action="store_true",
        default=False,
        help="Texture the ground and roofs with NAIP aerial photos (USGS, "
        "~0.6 m/px). Turns on --terrain. Off by default.",
    )
    parser.add_argument(
        "--min-quality",
        metavar="SCORE",
        type=float,
        default=None,
        help="Skip Mapillary images whose quality_score (0-1) is below SCORE, "
        "e.g. 0.6. Off by default.",
    )
    parser.add_argument(
        "--poses",
        choices=POSE_SOURCES,
        default="raw",
        help="Camera poses for texturing: Mapillary raw GPS (default), Mapillary computed, "
        "or COLMAP aligned to OSM (see src/pose_alignment/align_poses.py)",
    )
    parser.add_argument(
        "--aligned-only",
        action="store_true",
        help="With --poses colmap, texture only from images with an aligned pose",
    )
    parser.add_argument(
        "--alignment-dir",
        metavar="DIR",
        default=None,
        help="With --poses colmap, reuse this ROI folder (outputs/alignment/mapillary_roi_N)",
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
            "fill_obstructions": not args.no_fill,
            "output_path": args.output,
            "ask_output_path": ask_save_path,
            "mapillary_dump_path": args.dump_mapillary,
            "use_terrain": args.terrain or args.naip,
            "use_naip": args.naip,
            "min_quality": args.min_quality,
            "pose_source": args.poses,
            "aligned_only": args.aligned_only,
            "alignment_dir": args.alignment_dir,
        },
        profiler=profiler,
    )
    if profiler is not None:
        profiler.save(args.profile)


if __name__ == "__main__":
    main()
