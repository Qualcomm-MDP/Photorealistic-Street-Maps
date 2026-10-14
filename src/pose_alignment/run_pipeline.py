"""Run the alignment pipeline, from any stage.

Stages, in order (each is also its own script):
  gui           route_gui.py            draw the streets on a map -> ROI file
  mapillary     mapillary_layer.py      best Mapillary run + rig siblings -> data/mapillary/
  segmentation  segmentation_layer.py   Mask2Former building/road masks -> data/masks/
  colmap        colmap_layer.py         sparse models per leg and camera -> data/colmap/
  osm           osm_alignment_layer.py  aligned to OSM; corrected cameras -> data/alignment/
Each ROI has its own folder, rois/mapillary_roi_N/, holding data/ (the folders above) and
results/ (before/after overviews, per-leg alignments, COLMAP 3D views).

--from picks the first stage to run (default: auto). Without --roi-dir a new ROI starts at
the GUI (or at the Mapillary stage with --roi <file>). With --roi-dir, auto starts at the
first stage whose outputs are missing; an explicit --from checks that the earlier stages'
outputs exist and says which stage to start from if not. --to stops after a stage.

Finished work is reused (e.g. COLMAP models of legs already done, frames already
segmented), so an interrupted run can be resumed; --force redoes the stages that run.

Needs MAPILLARY_TOKEN for the mapillary stage; see README.md.

Usage:
  python run_pipeline.py                                        # draw a route, run everything
  python run_pipeline.py --roi routes/my_route.json             # skip the GUI
  python run_pipeline.py --roi-dir rois/mapillary_roi_1         # resume where it stopped
  python run_pipeline.py --roi-dir rois/mapillary_roi_1 --from colmap
  python run_pipeline.py --roi-dir rois/mapillary_roi_1 --from osm --camera-height 1.4
  python run_pipeline.py --view right --to segmentation
"""

import argparse
import json
import os
import time
from pathlib import Path

from common import ROIS_DIR, RoiPaths

STAGES = ["gui", "mapillary", "segmentation", "colmap", "osm"]


def stage_done(paths):
    """Which stages have complete outputs for this ROI folder."""
    from segmentation_layer import missing_masks

    done = {"gui": paths.roi_json.exists()}
    done["mapillary"] = paths.sequences_json.exists() and all(
        (paths.images / s["folder"] / "cameras.json").exists() for s in json.loads(paths.sequences_json.read_text()))
    done["segmentation"] = done["mapillary"] and not missing_masks(paths)
    legs = json.loads(paths.legs_json.read_text())["legs"] if paths.legs_json.exists() else []
    done["colmap"] = bool(legs) and all("model" in c for leg in legs for c in leg["cameras"])
    done["osm"] = paths.summary_json.exists()
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=["auto"] + STAGES, default="auto", help="First stage to run")
    parser.add_argument("--to", dest="stop", choices=STAGES, default="osm", help="Last stage to run")
    parser.add_argument("--roi-dir", type=Path, help="Existing ROI folder (rois/mapillary_roi_N) to continue")
    parser.add_argument("--roi", type=Path, help="ROI file from route_gui.py, to start at the mapillary stage")
    parser.add_argument("--rois-dir", type=Path, default=ROIS_DIR, help="Where new ROI folders are created")
    parser.add_argument("--force", action="store_true", help="Redo the stages that run instead of reusing their outputs")
    g = parser.add_argument_group("mapillary stage")
    g.add_argument("--view", choices=["forward", "right", "back", "left"], help="Only start from cameras looking this way")
    g.add_argument("--sequence", help="Start from this Mapillary sequence instead of the best one")
    g.add_argument("--min-images", type=int, default=20, help="Ignore runs with fewer frames")
    g.add_argument("--max-spacing", type=float, default=6.0, help="Prefer runs with frames at most this far apart (m)")
    g.add_argument("--max-images", type=int, help="Download at most this many frames per sequence")
    g.add_argument("--no-rig", action="store_true", help="Do not add the rig's other cameras")
    g.add_argument("--resolution", choices=["2048", "original"], default="2048")
    g.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MAPILLARY_ACCESS_TOKEN"))
    g = parser.add_argument_group("segmentation stage")
    g.add_argument("--previews", action="store_true", help="Also save colour mask overlays")
    g.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    g = parser.add_argument_group("colmap stage")
    g.add_argument("--max-turn", type=float, default=35.0, help="Split legs at turns sharper than this (deg)")
    g.add_argument("--trim-deg", type=float, default=20.0, help="Keep leg frames heading within this of the leg")
    g.add_argument("--min-frames", type=int, default=15, help="Skip legs with fewer primary frames")
    g.add_argument("--max-frames", type=int, default=60, help="Cut longer legs into chunks of at most this many frames")
    g.add_argument("--mask-moving", action="store_true", help="Ignore features on sky, vehicles and people")
    g.add_argument("--largest-piece", action="store_true", help="Use COLMAP's largest piece instead of a merged model")
    g.add_argument("--height-range", type=float, nargs=2, default=[0.9, 3.0], help="Plausible camera heights (m)")
    g = parser.add_argument_group("osm stage")
    g.add_argument("--camera-height", type=float, help="Camera height above the road (m); default: measured")
    g.add_argument("--max-tie-misfit", type=float, default=5.0, help="Drop a rig camera misfitting the others by more (m)")
    g.add_argument("--max-leg-error", type=float, default=2.0, help="Mark legs fitting the walls worse than this unreliable (m)")
    args = parser.parse_args()

    # Where to start.
    paths = RoiPaths(args.roi_dir) if args.roi_dir else None
    if paths and not paths.root.exists():
        raise SystemExit(f"{paths.root} does not exist")
    start = args.start
    if start == "auto":
        if paths is None:
            start = "mapillary" if args.roi else "gui"
        else:
            done = stage_done(paths)
            start = next((s for s in STAGES[1:] if not done[s]), None)
            if start is None:
                print(f"{paths.name}: every stage is complete (summary in {paths.summary_json}). "
                      "Use --from <stage> [--force] to redo one.")
                return
            print(f"{paths.name}: starting at the {start} stage (earlier stages are complete)")
    elif start not in ("gui", "mapillary"):
        if paths is None:
            raise SystemExit(f"--from {start} needs --roi-dir rois/mapillary_roi_N")
        done = stage_done(paths)
        for s in STAGES[1:STAGES.index(start)]:
            if not done[s]:
                raise SystemExit(f"{paths.name}: the {s} stage has not finished, so the pipeline cannot start at "
                                 f"{start}. Run with --from {s} (or leave --from out to resume automatically).")
    elif start == "mapillary" and not (args.roi or (paths and paths.roi_json.exists())):
        raise SystemExit("--from mapillary needs --roi <ROI file> (or --roi-dir with a roi.json); "
                         "otherwise start from the gui")
    if STAGES.index(start) > STAGES.index(args.stop):
        raise SystemExit(f"--to {args.stop} comes before the start stage ({start})")
    run = STAGES[STAGES.index(start):STAGES.index(args.stop) + 1]
    print("Stages: " + " -> ".join(run))
    timings = {}

    roi = None
    if "gui" in run:
        from route_gui import select_roi
        t0 = time.time()
        roi, _ = select_roi()
        timings["gui"] = time.time() - t0
    if "mapillary" in run:
        from mapillary_layer import fetch_roi
        if not args.token:
            raise SystemExit("Set MAPILLARY_TOKEN (https://www.mapillary.com/dashboard/developers) or pass --token")
        if roi is None:
            roi_file = args.roi if args.roi else paths.roi_json
            roi_file = roi_file / "roi.json" if roi_file.is_dir() else roi_file
            roi = json.loads(roi_file.read_text())
        t0 = time.time()
        # Always a new numbered folder, so an earlier ROI's results are never overwritten.
        paths = RoiPaths(fetch_roi(roi, args.token, None, args.view, sequence=args.sequence, min_images=args.min_images,
                                   max_spacing=args.max_spacing, max_images=args.max_images, no_rig=args.no_rig,
                                   resolution=args.resolution, rois_dir=args.rois_dir))
        timings["mapillary"] = time.time() - t0
    if paths is None:
        return
    if "segmentation" in run:
        from segmentation_layer import segment_roi
        t0 = time.time()
        segment_roi(paths.root, device=args.device, previews=args.previews, force=args.force)
        timings["segmentation"] = time.time() - t0
    if "colmap" in run:
        from colmap_layer import reconstruct_roi
        t0 = time.time()
        reconstruct_roi(paths.root, args.max_turn, args.trim_deg, args.min_frames, args.max_frames, args.mask_moving,
                        args.largest_piece, args.height_range, args.force)
        timings["colmap"] = time.time() - t0
    if "osm" in run:
        from osm_alignment_layer import align_roi
        t0 = time.time()
        align_roi(paths.root, args.camera_height, args.height_range, args.max_tie_misfit, args.max_leg_error)
        timings["osm"] = time.time() - t0

    print(f"\nROI folder: {paths.root}")
    print("Time: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timings.items()))
    if "osm" in run:
        print(f"Corrected cameras: {paths.alignment}/<sequence folder>.json; "
              f"images: {paths.results}")


if __name__ == "__main__":
    main()
