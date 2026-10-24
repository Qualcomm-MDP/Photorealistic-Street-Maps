"""Run the alignment pipeline, from any stage.

Stages, in order (each is also its own script):
  gui           route_gui.py            draw the streets on a map -> ROI file
  mapillary     mapillary_layer.py      best Mapillary run + rig siblings -> data/mapillary/
  segmentation  segmentation_layer.py   Mask2Former building/road masks -> data/masks/
  calibrate     calibrate.py            GeoCalib focal, distortion and per-frame gravity -> data/calibration/
  reconstruct   reconstruct.py          ALIKED + nearest-neighbour or LightGlue matching, COLMAP global
                                        mapper with gravity priors, per leg -> data/reconstruction/
  align         align.py                building points fitted to the OSM footprints; aligned cameras
                                        -> data/alignment/, figures -> results/
Each ROI has its own folder, rois/mapillary_roi_N/, holding data/ (the folders above) and
results/ (overview.png, fit_<leg>_<sequence>.png; blender/ with --blender).

--from picks the first stage to run (default: auto). Without --roi-dir a new ROI starts at
the GUI (or at the Mapillary stage with --roi <file>). With --roi-dir, auto starts at the
first stage whose outputs are missing; an explicit --from checks that the earlier stages'
outputs exist and says which stage to start from if not. --to stops after a stage.
The calibrate, reconstruct and align stages each run in their own process.

Needs MAPILLARY_TOKEN for the mapillary stage; see README.md.

Usage:
  python run_pipeline.py                                        # draw a route, run everything
  python run_pipeline.py --roi routes/my_route.json             # skip the GUI
  python run_pipeline.py --roi-dir rois/mapillary_roi_19        # resume where it stopped
  python run_pipeline.py --roi-dir rois/mapillary_roi_19 --from reconstruct --one-leg --blender
  python run_pipeline.py --roi-dir rois/mapillary_roi_19 --from reconstruct --tag test   # separate run
  python run_pipeline.py --view right --to segmentation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ROIS_DIR, RoiPaths

HERE = Path(__file__).resolve().parent
STAGES = ["gui", "mapillary", "segmentation", "calibrate", "reconstruct", "align"]


def stage_done(paths):
    """Which stages have complete outputs for this ROI folder."""
    from segmentation_layer import missing_masks

    done = {"gui": paths.roi_json.exists()}
    done["mapillary"] = paths.sequences_json.exists() and all(
        (paths.images / s["folder"] / "cameras.json").exists() for s in json.loads(paths.sequences_json.read_text()))
    done["segmentation"] = done["mapillary"] and not missing_masks(paths)
    done["calibrate"] = done["mapillary"] and all(paths.calibration_json(s["folder"]).exists() for s in paths.sequences())
    legs = json.loads(paths.legs_json.read_text())["legs"] if paths.legs_json.exists() else []
    done["reconstruct"] = bool(legs) and all("model" in c for leg in legs for c in leg["cameras"])
    done["align"] = paths.summary_json.exists()
    return done


def run_script(script, *args):
    """A stage in its own process (GeoCalib, PyTorch and COLMAP each bring their own libraries)."""
    r = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], cwd=HERE)
    if r.returncode:
        raise SystemExit(f"{script} failed ({r.returncode})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=["auto"] + STAGES, default="auto", help="First stage to run")
    parser.add_argument("--to", dest="stop", choices=STAGES, default="align", help="Last stage to run")
    parser.add_argument("--roi-dir", type=Path, help="Existing ROI folder (rois/mapillary_roi_N) to continue")
    parser.add_argument("--roi", type=Path, help="ROI file from route_gui.py, to start at the mapillary stage")
    parser.add_argument("--rois-dir", type=Path, default=ROIS_DIR, help="Where new ROI folders are created")
    parser.add_argument("--force", action="store_true", help="Redo segmentation and calibration instead of reusing them")
    parser.add_argument("--tag", help="Separate run of reconstruct and align: data/reconstruction_<tag>, "
                                      "data/alignment_<tag>, results/<tag> (the calibration is shared)")
    g = parser.add_argument_group("mapillary stage")
    g.add_argument("--view", choices=["forward", "right", "back", "left"], help="Only start from cameras looking this way")
    g.add_argument("--sequence", help="Start from this Mapillary sequence instead of the best one")
    g.add_argument("--min-images", type=int, default=20, help="Ignore runs with fewer frames")
    g.add_argument("--max-spacing", type=float, default=10.0, help="Prefer runs with frames at most this far apart (m)")
    g.add_argument("--max-images", type=int, help="Download at most this many frames per sequence")
    g.add_argument("--no-rig", action="store_true", help="Do not add the rig's other cameras")
    g.add_argument("--resolution", choices=["2048", "original"], default="2048")
    g.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MAPILLARY_ACCESS_TOKEN"))
    g = parser.add_argument_group("segmentation stage")
    g.add_argument("--previews", action="store_true", help="Also save colour mask overlays")
    g.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    g = parser.add_argument_group("reconstruct stage")
    g.add_argument("--max-turn", type=float, default=35.0, help="Split legs at turns sharper than this (deg)")
    g.add_argument("--trim-deg", type=float, default=20.0, help="Keep leg frames heading within this of the leg (deg)")
    g.add_argument("--min-frames", type=int, default=15, help="Skip legs with fewer primary frames")
    g.add_argument("--max-frames", type=int, default=60, help="Cut longer legs into chunks of at most this many frames")
    g.add_argument("--one-leg", action="store_true",
                   help="The whole route as one leg (--max-turn 180 --trim-deg 180 --max-frames 1000)")
    g.add_argument("--mapper", choices=["global", "incremental"], default="global",
                   help="COLMAP's global mapper (GLOMAP) with gravity priors, or its incremental mapper")
    g.add_argument("--split-weak", action="store_true",
                   help="Drop weakly anchored frames and fit each side of the gap separately (off by default)")
    g.add_argument("--matcher", choices=["auto", "nn", "lightglue"], default="auto",
                   help="auto: nearest-neighbour for frames under 6 m apart, LightGlue otherwise")
    g = parser.add_argument_group("align stage")
    g.add_argument("--max-error", type=float, default=2.0,
                   help="Reliable if the median point-to-footprint distance is at most this (m)")
    g.add_argument("--blender", action="store_true", help="Also export the aligned cameras for Blender (results/blender/)")
    args = parser.parse_args()
    if args.one_leg:
        args.max_turn, args.trim_deg, args.max_frames = 180.0, 180.0, 1000

    # Where to start.
    paths = RoiPaths(args.roi_dir, args.tag) if args.roi_dir else None
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
                      "Use --from <stage> to redo one.")
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
            roi_file = RoiPaths(roi_file).roi_json if roi_file.is_dir() else roi_file
            roi = json.loads(roi_file.read_text())
        t0 = time.time()
        # Always a new numbered folder, so an earlier ROI's results are never overwritten.
        paths = RoiPaths(fetch_roi(roi, args.token, None, args.view, sequence=args.sequence, min_images=args.min_images,
                                   max_spacing=args.max_spacing, max_images=args.max_images, no_rig=args.no_rig,
                                   resolution=args.resolution, rois_dir=args.rois_dir), args.tag)
        timings["mapillary"] = time.time() - t0
    if paths is None:
        return
    if "segmentation" in run:
        from segmentation_layer import segment_roi
        t0 = time.time()
        segment_roi(paths.root, device=args.device, previews=args.previews, force=args.force)
        timings["segmentation"] = time.time() - t0
    tag = ["--tag", args.tag] if args.tag else []
    if "calibrate" in run:
        t0 = time.time()
        run_script("calibrate.py", paths.root, *(["--force"] if args.force else []))
        timings["calibrate"] = time.time() - t0
    if "reconstruct" in run:
        t0 = time.time()
        run_script("reconstruct.py", paths.root, "--max-turn", args.max_turn, "--trim-deg", args.trim_deg,
                   "--min-frames", args.min_frames, "--max-frames", args.max_frames, "--matcher", args.matcher,
                   "--mapper", args.mapper, *tag,
                   *(["--split-weak"] if args.split_weak else []))
        timings["reconstruct"] = time.time() - t0
    if "align" in run:
        t0 = time.time()
        run_script("align.py", paths.root, "--max-error", args.max_error, *tag)
        if args.blender:
            run_script("export_blender.py", paths.root, *tag)
        timings["align"] = time.time() - t0

    print(f"\nROI folder: {paths.root}")
    print("Time: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timings.items())
          + f"; total {sum(timings.values()) / 60:.1f} min")
    if "align" in run:
        print(f"Aligned cameras: {paths.alignment}/<sequence folder>.json; figures: {paths.results}")


if __name__ == "__main__":
    main()
