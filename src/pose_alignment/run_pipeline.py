"""Run the alignment pipeline, from any stage.

Stages, in order (each is also its own script):
  gui           route_gui.py            draw the streets on a map -> ROI file
  mapillary     data_ingest/mapillary_sequences.py   best Mapillary run + rig siblings -> data/mapillary/
                                        (a 360 run: right and left pinhole crops, as a two-camera rig)
  segmentation  segmentation/street_masks.py   Mask2Former building/road masks -> data/masks/
  calibrate     calibrate.py            GeoCalib focal, distortion and per-frame gravity -> data/calibration/
                                        (--calibration mapillary: Mapillary's -> data/calibration_mapillary/)
  reconstruct   reconstruct.py          ALIKED + nearest-neighbour or LightGlue matching, COLMAP global
                                        mapper with gravity priors, per leg -> data/reconstruction/
  align         align.py                building points fitted to the OSM footprints; aligned cameras
                                        -> data/alignment/, figures -> results/
Each ROI has its own folder, outputs/alignment/mapillary_roi_N/, holding data/ (the folders above) and
results/ (overview.png, fit_<leg>_<sequence>.png).

--from picks the first stage to run (default: auto). Without --roi-dir a new ROI starts at
the GUI (or at the Mapillary stage with --roi <file>). With --roi-dir, auto starts at the
first stage whose outputs are missing; an explicit --from checks that the earlier stages'
outputs exist and says which stage to start from if not. --to stops after a stage.
The segmentation, calibrate, reconstruct and align stages each run in their own process.

Needs MAPILLARY_TOKEN for the mapillary stage; see README.md.

Usage:
  python -m pose_alignment.run_pipeline                                        # draw a route, run everything
  python -m pose_alignment.run_pipeline --roi outputs/alignment/routes/my_route.json             # skip the GUI
  python -m pose_alignment.run_pipeline --roi-dir outputs/alignment/mapillary_roi_19        # resume where it stopped
  python -m pose_alignment.run_pipeline --roi-dir outputs/alignment/mapillary_roi_19 --from reconstruct --one-leg
  python -m pose_alignment.run_pipeline --roi-dir outputs/alignment/mapillary_roi_21 --calibration mapillary --tag mly    # wide rig lenses
  python -m pose_alignment.run_pipeline --roi-dir outputs/alignment/mapillary_roi_19 --from reconstruct --tag test   # separate run
  python -m pose_alignment.run_pipeline --view right --to segmentation
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from .geo_common import ROIS_DIR, RoiPaths, module_command

STAGES = ["gui", "mapillary", "segmentation", "calibrate", "reconstruct", "align"]


def stage_done(paths):
    """Which stages have complete outputs for this ROI folder."""
    from segmentation.street_masks import missing_masks

    done = {"gui": paths.roi_json.exists()}
    done["mapillary"] = paths.sequences_json.exists() and all(
        (paths.images / s["folder"] / "cameras.json").exists()
        for s in json.loads(paths.sequences_json.read_text())
    )
    done["segmentation"] = done["mapillary"] and not missing_masks(paths)
    done["calibrate"] = done["mapillary"] and all(
        paths.calibration_json(s["folder"]).exists() for s in paths.sequences()
    )
    doc = json.loads(paths.legs_json.read_text()) if paths.legs_json.exists() else {}
    legs = (
        doc.get("legs", [])
        if doc.get("calibration", "geocalib") == paths.calibration_source
        else []
    )
    done["reconstruct"] = bool(legs) and all(
        "model" in c for leg in legs for c in leg["cameras"]
    )
    done["align"] = paths.summary_json.exists()
    return done


def report_timings(paths, timings, parts):
    """Per stage and part: seconds and share of the pipeline's own time (the GUI is the user's
    drawing time, listed but not counted). Also written to results[/<tag>]/timings.json."""
    work = {k: v for k, v in timings.items() if k != "gui"}
    total = sum(work.values()) or 1e-9
    frames = (
        sum(len(paths.frames(s["folder"])) for s in paths.sequences())
        if paths.sequences_json.exists()
        else 0
    )
    print(f"\nTime per stage ({frames} frames):")
    for stage, sec in timings.items():
        share = "" if stage == "gui" else f"{100 * sec / total:5.1f}%"
        note = "  (your drawing time; not counted)" if stage == "gui" else ""
        print(f"  {stage:<14} {sec / 60:6.2f} min {share}{note}")
        for part, s in (parts.get(stage) or {}).items():
            print(f"      {part:<38} {s:7.1f} s")
    print(
        f"  {'total':<14} {total / 60:6.2f} min"
        + (f"  ({total / frames:.1f} s per frame)" if frames else "")
    )
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "timings.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "total_s": round(total, 1),
                "stages_s": {k: round(v, 1) for k, v in timings.items()},
                "parts_s": {
                    k: {p: round(s, 1) for p, s in v.items()} for k, v in parts.items()
                },
            },
            indent=1,
        )
    )


def run_script(script, *args):
    """A stage in its own process (GeoCalib, PyTorch and COLMAP each bring their own libraries)."""
    cmd, env = module_command(script, *args)
    r = subprocess.run(cmd, env=env)
    if r.returncode:
        raise SystemExit(f"{script} failed ({r.returncode})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from",
        dest="start",
        choices=["auto"] + STAGES,
        default="auto",
        help="First stage to run",
    )
    parser.add_argument(
        "--to", dest="stop", choices=STAGES, default="align", help="Last stage to run"
    )
    parser.add_argument(
        "--roi-dir",
        type=Path,
        help="Existing ROI folder (outputs/alignment/mapillary_roi_N) to continue",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        help="ROI file from route_gui.py, to start at the mapillary stage",
    )
    parser.add_argument(
        "--rois-dir",
        type=Path,
        default=ROIS_DIR,
        help="Where new ROI folders are created",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo segmentation and calibration instead of reusing them",
    )
    parser.add_argument(
        "--tag",
        help="Separate run of reconstruct and align: data/reconstruction_<tag>, "
        "data/alignment_<tag>, results/<tag> (the calibration is shared)",
    )
    g = parser.add_argument_group("mapillary stage")
    g.add_argument(
        "--out",
        type=Path,
        help="Folder for the new ROI (default: the next free mapillary_roi_N in --rois-dir)",
    )
    g.add_argument(
        "--images",
        type=Path,
        help="JSON list of Mapillary records (ALIGNMENT_FIELDS) to pick the captures from, "
        "instead of searching Mapillary (the main pipeline's fetch stage writes it)",
    )
    g.add_argument(
        "--selection",
        type=Path,
        help="JSON selection of captures (data_ingest.mapillary_sequences.choose_roi) to "
        "download instead of choosing them (the main pipeline's fetch stage writes it)",
    )
    g.add_argument(
        "--one-run",
        action="store_true",
        help="Only the best run per line (by default further runs cover where it ends)",
    )
    g.add_argument(
        "--keep-originals",
        action="store_true",
        help="Download full-resolution photos into data/mapillary/originals/ and scale the "
        "frames from them (the main pipeline's texturing reuses them)",
    )
    g.add_argument(
        "--view",
        choices=["forward", "right", "back", "left"],
        help="Only start from cameras looking this way",
    )
    g.add_argument(
        "--sequence", help="Start from this Mapillary sequence instead of the best one"
    )
    g.add_argument(
        "--min-images", type=int, default=20, help="Ignore runs with fewer frames"
    )
    g.add_argument(
        "--max-spacing",
        type=float,
        default=10.0,
        help="Prefer runs with frames at most this far apart (m)",
    )
    g.add_argument(
        "--max-images", type=int, help="Download at most this many frames per sequence"
    )
    g.add_argument(
        "--no-rig", action="store_true", help="Do not add the rig's other cameras"
    )
    g.add_argument(
        "--resolution",
        choices=["2048", "original"],
        default="2048",
        help="Perspective frames' size (360 frames are always downloaded at full size)",
    )
    g.add_argument(
        "--pano",
        choices=["auto", "prefer", "off"],
        default="auto",
        help="360 runs, cut into right and left views: taken when they cover more of the line (auto), "
        "whenever usable (prefer), or never (off)",
    )
    g.add_argument(
        "--pano-fov",
        type=float,
        default=90.0,
        help="Field of view of the 360 crops (deg, square)",
    )
    g.add_argument(
        "--token",
        default=os.environ.get("MAPILLARY_TOKEN")
        or os.environ.get("MAPILLARY_ACCESS_TOKEN"),
    )
    g = parser.add_argument_group("segmentation stage")
    g.add_argument(
        "--previews", action="store_true", help="Also save colour mask overlays"
    )
    g.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    g = parser.add_argument_group("calibrate stage")
    g.add_argument(
        "--calibration",
        choices=["geocalib", "mapillary"],
        default="geocalib",
        help="Focal and distortion from GeoCalib (default) or from Mapillary (data/calibration_mapillary/), "
        "for lenses GeoCalib warns about; gravity from GeoCalib either way",
    )
    g.add_argument(
        "--gravity-every",
        type=int,
        default=5,
        help="Estimate gravity on every n-th frame and interpolate between (1: every frame)",
    )
    g = parser.add_argument_group("reconstruct stage")
    g.add_argument(
        "--fix-intrinsics",
        action="store_true",
        help="Keep the calibration's focal and distortion fixed instead of refining them",
    )
    g.add_argument(
        "--fix-focal",
        action="store_true",
        help="Keep the focal fixed but refine the distortion",
    )
    g.add_argument(
        "--focal",
        type=float,
        help="Focal as a fraction of the long side, instead of GeoCalib's",
    )
    g.add_argument(
        "--max-turn",
        type=float,
        default=35.0,
        help="Split legs at turns sharper than this (deg)",
    )
    g.add_argument(
        "--trim-deg",
        type=float,
        default=20.0,
        help="Keep leg frames heading within this of the leg (deg)",
    )
    g.add_argument(
        "--min-frames", type=int, default=15, help="Skip legs with fewer primary frames"
    )
    g.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Cut longer legs into chunks of at most this many frames",
    )
    g.add_argument(
        "--leg-overlap",
        type=int,
        default=8,
        help="Run each leg on into the next one's first frames, which tie the legs together (0: off)",
    )
    g.add_argument(
        "--no-shared-focal",
        action="store_true",
        help="Keep each leg's own refined focal (by default every leg is solved again with one focal per camera)",
    )
    g.add_argument(
        "--solo-cameras",
        action="store_true",
        help="Solve each camera of a rig on its own (by default a leg's rig cameras are solved together)",
    )
    g.add_argument(
        "--rig", action="store_true", help=argparse.SUPPRESS
    )  # the default now; kept for old commands
    g.add_argument(
        "--gap-factor",
        type=float,
        default=3.0,
        help="Cut each side or rear camera of a leg to its longest stretch without a jump over this x its "
        "frame spacing (0: off)",
    )
    g.add_argument(
        "--one-leg",
        action="store_true",
        help="The whole route as one leg (--max-turn 180 --trim-deg 180 --max-frames 1000)",
    )
    g.add_argument(
        "--mapper",
        choices=["global", "incremental"],
        default="global",
        help="COLMAP's global mapper (GLOMAP) with gravity priors, or its incremental mapper",
    )
    g.add_argument(
        "--split-weak",
        action="store_true",
        help="Drop weakly anchored frames and fit each side of the gap separately (off by default)",
    )
    g.add_argument(
        "--matcher",
        choices=["auto", "nn", "lightglue"],
        default="auto",
        help="auto: nearest-neighbour for frames under 6 m apart, LightGlue otherwise",
    )
    g = parser.add_argument_group("align stage")
    g.add_argument(
        "--max-error",
        type=float,
        default=2.0,
        help="Reliable if the median point-to-footprint distance is at most this (m)",
    )
    g.add_argument(
        "--no-link",
        action="store_true",
        help="Fit each leg on its own instead of tying linked legs together",
    )
    args = parser.parse_args()
    if args.one_leg:
        args.max_turn, args.trim_deg, args.max_frames = 180.0, 180.0, 1000

    # Where to start.
    paths = RoiPaths(args.roi_dir, args.tag, args.calibration) if args.roi_dir else None
    if paths and not paths.root.exists():
        raise SystemExit(f"{paths.root} does not exist")
    start = args.start
    if start == "auto":
        if paths is None:
            start = "mapillary" if args.roi or args.selection else "gui"
        else:
            done = stage_done(paths)
            start = next((s for s in STAGES[1:] if not done[s]), None)
            if start is None:
                print(
                    f"{paths.name}: every stage is complete (summary in {paths.summary_json}). "
                    "Use --from <stage> to redo one."
                )
                return
            print(
                f"{paths.name}: starting at the {start} stage (earlier stages are complete)"
            )
    elif start not in ("gui", "mapillary"):
        if paths is None:
            raise SystemExit(
                f"--from {start} needs --roi-dir outputs/alignment/mapillary_roi_N"
            )
        done = stage_done(paths)
        for s in STAGES[1 : STAGES.index(start)]:
            if not done[s]:
                raise SystemExit(
                    f"{paths.name}: the {s} stage has not finished, so the pipeline cannot start at "
                    f"{start}. Run with --from {s} (or leave --from out to resume automatically)."
                )
    elif start == "mapillary" and not (
        args.roi or args.selection or (paths and paths.roi_json.exists())
    ):
        raise SystemExit(
            "--from mapillary needs --roi <ROI file> (or --roi-dir with a roi.json); "
            "otherwise start from the gui"
        )
    if STAGES.index(start) > STAGES.index(args.stop):
        raise SystemExit(f"--to {args.stop} comes before the start stage ({start})")
    run = STAGES[STAGES.index(start) : STAGES.index(args.stop) + 1]
    print("Stages: " + " -> ".join(run))
    timings, parts = {}, {}  # stage -> seconds; stage -> {part: seconds}

    roi = None
    if "gui" in run:
        from .route_gui import select_roi

        t0 = time.time()
        roi, _ = select_roi()
        timings["gui"] = time.time() - t0
    if "mapillary" in run:
        from data_ingest.mapillary_sequences import fetch_roi

        if not args.token and not (args.images or args.selection):
            raise SystemExit(
                "Set MAPILLARY_TOKEN (https://www.mapillary.com/dashboard/developers) or pass --token"
            )
        if roi is None and not args.selection:
            roi_file = args.roi if args.roi else paths.roi_json
            roi_file = RoiPaths(roi_file).roi_json if roi_file.is_dir() else roi_file
            roi = json.loads(roi_file.read_text())
        t0 = time.time()
        # A new folder (by default the next numbered one), so an earlier ROI's results are
        # never overwritten.
        if args.selection:
            from data_ingest.mapillary_sequences import download_selection

            roi_dir = download_selection(
                json.loads(args.selection.read_text()),
                args.out,
                args.max_images,
                args.resolution,
                args.rois_dir,
                args.pano_fov,
                args.keep_originals,
            )
        else:
            roi_dir = fetch_roi(
                roi,
                args.token,
                args.out,
                args.view,
                sequence=args.sequence,
                min_images=args.min_images,
                max_spacing=args.max_spacing,
                max_images=args.max_images,
                no_rig=args.no_rig,
                resolution=args.resolution,
                rois_dir=args.rois_dir,
                pano=args.pano,
                pano_fov=args.pano_fov,
                images=json.loads(args.images.read_text()) if args.images else None,
                keep_originals=args.keep_originals,
                one_run=args.one_run,
            )
        paths = RoiPaths(roi_dir, args.tag, args.calibration)
        timings["mapillary"] = time.time() - t0
        from data_ingest.mapillary_sequences import TIMINGS

        parts["mapillary"] = {
            "search Mapillary": TIMINGS["search_s"],
            f"download {TIMINGS['frames']} frames": TIMINGS["download_s"],
        }
    if paths is None:
        return
    if "segmentation" in run:
        t0 = time.time()
        # In its own process like the later stages: PyTorch's threads left running in this one made a later
        # stage's launch (fork) abort now and then on macOS (malloc "Internal invariant broken", -6).
        run_script(
            "segmentation.street_masks",
            paths.root,
            *(["--device", args.device] if args.device else []),
            *(["--previews"] if args.previews else []),
            *(["--force"] if args.force else []),
        )
        timings["segmentation"] = time.time() - t0
    tag = ["--tag", args.tag] if args.tag else []
    if "calibrate" in run:
        t0 = time.time()
        run_script(
            "calibrate.py",
            paths.root,
            "--calibration",
            args.calibration,
            "--gravity-every",
            args.gravity_every,
            *(["--force"] if args.force else []),
        )
        timings["calibrate"] = time.time() - t0
        fresh = [
            json.loads(f.read_text())
            for f in paths.calibration.glob("*.json")
            if f.stat().st_mtime >= t0
        ]
        if fresh:
            parts["calibrate"] = {
                "focal and distortion (sampled frames)": sum(
                    d["seconds"]["calibration"] for d in fresh
                ),
                "gravity, every frame": sum(d["seconds"]["gravity"] for d in fresh),
            }
    if "reconstruct" in run:
        t0 = time.time()
        run_script(
            "reconstruct.py",
            paths.root,
            "--max-turn",
            args.max_turn,
            "--trim-deg",
            args.trim_deg,
            "--min-frames",
            args.min_frames,
            "--max-frames",
            args.max_frames,
            "--matcher",
            args.matcher,
            "--gap-factor",
            args.gap_factor,
            *(["--solo-cameras"] if args.solo_cameras else []),
            "--leg-overlap",
            args.leg_overlap,
            *(["--no-shared-focal"] if args.no_shared_focal else []),
            "--mapper",
            args.mapper,
            "--calibration",
            args.calibration,
            *tag,
            *(["--fix-intrinsics"] if args.fix_intrinsics else []),
            *(["--fix-focal"] if args.fix_focal else []),
            *(["--focal", args.focal] if args.focal else []),
            *(["--split-weak"] if args.split_weak else []),
        )
        timings["reconstruct"] = time.time() - t0
        if paths.legs_json.exists():
            names = {
                "prepare": "prepare frames and masks",
                "features_matching": "COLMAP import and match check (CPU)",
                "mapper": "COLMAP mapper",
                "mapper_fallback": "incremental-mapper fallback",
                "retriangulation": "re-triangulation",
            }
            ldoc = json.loads(paths.legs_json.read_text())
            parts["reconstruct"] = {
                "keypoints and matching (GPU, one process)": (
                    ldoc.get("seconds") or {}
                ).get("features_and_matching", 0.0)
            }
            for leg in ldoc["legs"]:
                for cam in leg["cameras"]:
                    for k, v in (cam.get("seconds") or {}).items():
                        if k != "total":
                            parts["reconstruct"][names.get(k, k)] = (
                                parts["reconstruct"].get(names.get(k, k), 0.0) + v
                            )
    if "align" in run:
        t0 = time.time()
        run_script(
            "align.py",
            paths.root,
            "--max-error",
            args.max_error,
            *tag,
            *(["--no-link"] if args.no_link else []),
        )
        timings["align"] = time.time() - t0

    print(f"\nROI folder: {paths.root}")
    report_timings(paths, timings, parts)
    if "align" in run:
        print(
            f"Aligned cameras: {paths.alignment}/<sequence folder>.json; figures: {paths.results}"
        )


if __name__ == "__main__":
    main()
