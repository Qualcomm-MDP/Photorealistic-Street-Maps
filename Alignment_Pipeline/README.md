# Alignment Pipeline

Corrects the positions and orientations of Mapillary street-view cameras by reconstructing
the street with COLMAP and fitting the reconstruction onto OpenStreetMap building
footprints. You draw the streets on a map; the pipeline fetches the imagery, segments
buildings and roads, builds sparse point clouds, matches the facades in them to the
street-facing walls in OSM, and writes corrected camera poses.

Raw Mapillary GPS is typically 5–20 m off on a street like this, and Mapillary's own
OpenSfM positions can be rotated about 10° relative to the street. After alignment, the
reconstructed facades typically sit within 0.2–0.7 m of the OSM walls (median).

## Stages

| # | Stage | Script | Input | Output |
|---|-------|--------|-------|--------|
| 1 | gui | `route_gui.py` | — | ROI file: the drawn lines, corridor width and bounding box (`routes/*.json`) |
| 2 | mapillary | `mapillary_layer.py` | ROI file | `data/mapillary/`: one folder per sequence (`frame_0001.jpg` …, `cameras.json`), `roi.json`, `sequences.json` |
| 3 | segmentation | `segmentation_layer.py` | ROI folder | `data/masks/`: a building/road mask per frame (optional colour previews) |
| 4 | colmap | `colmap_layer.py` | ROI folder + masks | `data/colmap/`: sparse models per leg and camera, `legs.json` |
| 5 | osm | `osm_alignment_layer.py` | COLMAP models | `data/alignment/`: corrected cameras per sequence (`<sequence folder>.json`), `summary.json` |

`run_pipeline.py` runs them in order, starting from any stage. `common.py` holds the shared
helpers (geography, ROI folders, COLMAP readers, geometry), and `alignment_plots.py` draws the
OSM stage's images.

Each ROI gets its own folder, `rois/mapillary_roi_N/`. The output folders in the table
above are inside its `data/` folder, and the result images are collected in its `results/`
folder:

| Image | Shows |
|-------|-------|
| `alignment_overview.png` | Before/after for the whole ROI |
| `raw_gps_vs_pipeline.png` | Raw GPS against the corrected camera positions |
| `legs/<leg>_alignment.png` | Before/after for one leg |
| `sparse_views/<leg>_<sequence folder>.png` | 3D view of one COLMAP model |

### 1. Draw the route (`route_gui.py`)

Opens a map in your browser, served by a small local web server. Click along the streets
you want; turns are fine. **New street** starts a separate line. Set the corridor
half-width: it must cover your clicking error but should not reach parallel streets
(15 m works for most roads). **Use this route** saves the ROI and hands it to the next
stage.

### 2. Fetch the imagery (`mapillary_layer.py`)

For each drawn line, Mapillary is searched in small tiles along it, because a large
bounding-box query returns only a sample of the images. Images outside the corridor and
360° panoramas are dropped. Each sequence (one capture session of one camera) is split
into continuous runs at time gaps and position jumps. The best run is picked in this
order:

1. It has enough frames.
2. Its frames are dense enough for COLMAP: a median spacing of at most 6 m
   (`--max-spacing`). In testing, a 7 m run split in COLMAP and ended 3.3 m off the walls.
3. It covers the largest share of the drawn line.
4. It has the densest spacing, then the most frames.

On a street where no single dense capture covers the whole line, only part of the line is
aligned.
`--view right` keeps only cameras looking right of the travel direction, and
`--sequence ID` forces a sequence.

Rig siblings are added: vehicles with several cameras upload one sequence per camera, and
seeing both sides of the street is what fixes the reconstruction's scale. Each sequence is
downloaded to `rois/mapillary_roi_N/data/mapillary/line<L>_cam<k>_<id>/`. If
`mapillary_roi_1` already exists, the next free number is used.

Each sequence folder's `cameras.json` holds, per frame: the Mapillary image id, capture
time, raw GPS, Mapillary's computed position and rotation (OpenSfM), the compass heading,
altitude, and the camera's make, model and calibration.

### 3. Segment (`segmentation_layer.py`)

Mask2Former (Swin-Large trained on Mapillary Vistas) labels every pixel. The 65 classes
are reduced to four values per mask PNG: 0 for other, 100 for road, 200 for building, and
255 for sky and moving things. The masks are always saved because the later stages read
them. `--previews` also saves colour overlays for checking by eye. It runs at about 0.3 s
per frame on an Apple-silicon GPU (MPS) and is slower on a CPU.

### 4. Reconstruct (`colmap_layer.py`)

- **Legs.** COLMAP cannot follow sparse street imagery around a corner, so each line's
  capture is split into straight legs at turns over 35°. The curved frames at the corner
  are trimmed, and long legs are cut into chunks of at most 60 frames at full density.
- **COLMAP per leg and camera.**
  - SIFT features, with Mapillary's calibration held fixed.
  - Sequential matching.
  - The incremental mapper, with thresholds relaxed for forward motion.
  - If the mapper splits a leg into pieces, they are merged.
- **Feature masks.**
  - Burned-in overlays such as timestamps and bonnets are detected and ignored.
  - `--mask-moving` also ignores sky, vehicles and people. This is off by default because
    the pipeline was tuned without it.
- **Point labels.** Each 3D point is labelled building or road from the masks.
- **Calibration check.** The camera's height above the road plane flags a bad Mapillary
  focal length. For forward-facing cameras, a self-calibrated rerun is then compared
  against OSM, and the better fit is kept.
- **Output.** `rois/mapillary_roi_N/results/sparse_views/` has a 3D view of each model, plus
  a plan view coloured by label.

### 5. Align to OSM (`osm_alignment_layer.py`)

`OSMAlignmentLayer.align()` aligns one leg, with one method per step (`_level_up_vector`,
`_tie_rig_cameras`, `_mapillary_only`, `_extract_facades_ransac`, `_get_nearby_osm_walls`,
`_fit_2d_similarity`, `_format_output`); `align_roi()` runs every leg of an ROI and writes
the outputs. For each leg, the steps are:

1. **Level the model.** Building walls are vertical and the road is roughly flat, so the
   reconstruction is turned upright using its own walls and road.
2. **Tie the rig's cameras together.** They are matched by capture time, and a camera
   whose path disagrees by more than 5 m is dropped.
3. **Extract facades.** Building points are projected to plan view and reduced to facade
   line segments by RANSAC.
4. **Fit to OSM.** The facades are matched to the street-facing walls of the OSM
   footprints by a global search over heading, scale and position. A robust refinement
   then pulls each facade onto its wall's line.

The scale is left free when facades are seen on both sides of the street. When they are
seen on one side only, the scale is held to Mapillary's track.

Outputs:

- **`<sequence folder>.json`** (in `rois/mapillary_roi_N/data/alignment/`) has one entry per
  frame:
  - `pipeline`: lat/lon, height above the road, `enu_m` (metres east/north/up from
    `enu_origin_lon_lat`), `rotation_cam_to_enu` (OpenCV camera axes in ENU),
    `blender_matrix` (the same rotation for a Blender camera), heading/pitch/roll, and
    `intrinsics` (COLMAP calibration of the downloaded image);
  - `raw_gps` and `mapillary_computed`: each with its offset to the pipeline position;
  - `not_aligned_reason`, for frames that could not be aligned (corner frames, frames
    COLMAP did not register, dropped cameras);
  - `reliable`: true when the frame's leg ends up within 2 m of the OSM walls (median
    facade-to-wall, `--max-leg-error`) and closer than with Mapillary's positions alone.
    Streets with few street-facing OSM walls, such as parking lots or set-back buildings,
    can fail to align; their cameras are kept but marked `reliable: false`.
- **`results/alignment_overview.png`**: before/after for the whole ROI. The left panel shows the
  reconstruction placed with Mapillary's positions only; the right panel shows it after
  matching facades to OSM walls.
- **`results/raw_gps_vs_pipeline.png`**: raw GPS against corrected
  camera positions and headings.
- **`results/legs/<leg>_alignment.png`**: the same before/after for each
  leg.
- **`summary.json`**: the numbers and any flags.

## Running

```bash
python run_pipeline.py                                   # draw a route, then run everything
python run_pipeline.py --roi routes/my_route.json        # reuse a drawn route
python run_pipeline.py --roi-dir rois/mapillary_roi_1    # resume: starts at the first unfinished stage
python run_pipeline.py --roi-dir rois/mapillary_roi_1 --from colmap   # rerun from COLMAP
python run_pipeline.py --roi-dir rois/mapillary_roi_1 --from osm --camera-height 1.4
python run_pipeline.py --view right --to segmentation    # stop after a stage
```

- **`--from`**: `gui`, `mapillary`, `segmentation`, `colmap` or `osm`. The default
  (`auto`) starts at the GUI for a new ROI, or at the first unfinished stage of
  `--roi-dir`. An explicit `--from` checks that the earlier stages' outputs exist, and
  tells you which stage to start from if they don't.
- **Reuse.** Finished work, such as segmented frames and the COLMAP models of finished
  legs, is reused. `--force` redoes the stages that run.
- **The mapillary stage** always writes a new `mapillary_roi_N` folder.

Every stage script also runs on its own:

```bash
python route_gui.py
python mapillary_layer.py --roi routes/my_route.json [--view right] [--list]
python segmentation_layer.py rois/mapillary_roi_1 [--previews]
python colmap_layer.py rois/mapillary_roi_1 [--mask-moving]
python osm_alignment_layer.py rois/mapillary_roi_1
```

`python <script> --help` lists every option.

### Time

COLMAP runs on the CPU (Homebrew COLMAP has no CUDA) and dominates the run time.

| Run (measured on an Apple-silicon laptop) | Time |
|-----|------|
| East Huron: one camera, 36 frames, 1 leg | 2.3 min |
| East Madison and Thompson with a turn: two rig cameras, 323 frames, 4 legs | 10.3 min (COLMAP 7.4) |

## Setup

1. Python 3.10+ and the packages:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The first segmentation run downloads the Mask2Former weights (about 900 MB) from Hugging
   Face.
2. COLMAP 3.9+ on the PATH (`brew install colmap` on macOS; see
   https://colmap.github.io/install.html).
3. A Mapillary client access token from https://www.mapillary.com/dashboard/developers:
   ```bash
   export MAPILLARY_TOKEN='MLY|...'
   ```
   Do not commit the token.

OSM data comes from the public Overpass API (no key needed). The GUI's basemap is
OpenStreetMap's own tile server by default, with OSM Humanitarian, OpenTopoMap and Esri
aerial in the layer switcher. None of these needs an API key.

## Data layout

```
routes/
  my_route_20261007_181500.json          drawn route (route_gui.py)
rois/
  mapillary_roi_1/                       one parent folder per ROI
    data/
      mapillary/
        roi.json  sequences.json
        line1_cam0_dIwbVYTq/  frame_0001.jpg ...  cameras.json
        line1_cam1_iMuXP3C2/  ...
      masks/
        classes.json  line1_cam0_dIwbVYTq/frame_0001.png ...
      colmap/
        legs.json  leg01/line1_cam0_dIwbVYTq/{sparse/, colmap.log}
      alignment/
        line1_cam0_dIwbVYTq.json  line1_cam1_iMuXP3C2.json  summary.json  legs/leg01.json ...
    results/
      alignment_overview.png  raw_gps_vs_pipeline.png
      legs/leg01_alignment.png ...
      sparse_views/leg01_line1_cam0_dIwbVYTq.png ...
  mapillary_roi_2/
    ...
```

`rois/*/data/` is in `.gitignore`. Each ROI's `results/` is not, so example images can be
committed. A whole ROI folder can be moved, renamed or deleted on its own.
