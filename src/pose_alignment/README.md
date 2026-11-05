# Alignment Pipeline

Places Mapillary street-view cameras, and a sparse point cloud of the street, on
OpenStreetMap. You draw the streets on a map; the pipeline fetches the imagery, segments
buildings and roads, takes each camera's calibration and gravity from Mapillary, reconstructs each stretch with
COLMAP, and fits the reconstructed building points onto the OSM building footprints. GPS is
used only as a starting point.

Raw Mapillary GPS is typically 5–20 m off. After alignment, the building points of a
reliable leg typically sit 0.5–1 m (median) from the OSM footprint edges.

## Stages

| #   | Stage        | Script                               | Output (in the ROI folder)                                                                    |
| --- | ------------ | ------------------------------------ | --------------------------------------------------------------------------------------------- |
| 1   | gui          | `route_gui.py`                       | the drawn route (`outputs/alignment/routes/*.json`)                                           |
| 2   | mapillary    | `data_ingest/mapillary_sequences.py` | `data/mapillary/`: frames and `cameras.json` per sequence, `roi.json`, `sequences.json`       |
| 3   | segmentation | `segmentation/street_masks.py`       | `data/masks/`: building/road/sky-and-moving mask per frame                                    |
| 4   | calibrate    | `calibrate.py`                       | `data/calibration/<sequence>.json`: focal, distortion, per-frame gravity (all from Mapillary) |
| 5   | reconstruct  | `reconstruct.py`                     | `data/reconstruction/`: COLMAP model per leg and sequence, `legs.json`                        |
| 6   | align        | `align.py`                           | `data/alignment/`: aligned cameras per sequence, `summary.json`; figures in `results/`        |

The mapillary and segmentation stages live with the main pipeline's own stages of the same
kind (`src/data_ingest/`, `src/segmentation/`); the others are in this folder.
`run_pipeline.py` runs them in order from any stage. Helpers: `geo_common.py` (geography, ROI
folders, COLMAP readers), `osm.py` (OSM buildings), `learned_features.py` and `nn_features.py`
(feature matching), `plot_fit.py` (figures). Tools: `gravity_check.py` (how consistent each
gravity source is with the reconstructed cameras), `shape_check.py` (how straight and
drift-free each reconstruction is). To see the aligned photos on the buildings, run the main
pipeline with `--poses colmap`: texturing projects them and the web app's 3D viewer shows the
.glb (Aadi's `scripts/eval/` renders it from the photos' poses in Blender).

## Running

### In the main pipeline

`src/main.py --poses colmap` runs this pipeline on the OSM streets in the bounding box
(`streets_from_bbox.py`) and textures with the aligned poses (`align_poses.py`); see that
file's docstring for what happens to images the pipeline does not align. The stages share
their work with the main pipeline's:

- **Images:** the fetch stage (`data_ingest/ingest.py`) asks Mapillary for the fields this
    pipeline needs too, over the box grown by 70 m, and the captures are picked from those
    records instead of a second search. Each photo is downloaded once, at full resolution,
    into the ROI's `data/mapillary/originals/` (`data_ingest/image_store.py`); the frames are
    scaled from it to 2048 px, and texturing uses the stored photo instead of downloading it.
- **Segmentation:** one Mask2Former checkpoint (Mapillary Vistas, `segmentation/model.py`)
    serves both. The segmentation stage keeps each frame's class map in `data/labels/`, and
    texturing's obstruction removal uses those maps for the same photos instead of
    segmenting them again.

### On its own

The scripts are a package inside `src/`, so run them as modules from the repository root
with `src` on the import path (`export PYTHONPATH=src`). ROI folders go to
`outputs/alignment/`.

```bash
export MAPILLARY_ACCESS_TOKEN='MLY|...'                # for the mapillary stage
python -m pose_alignment.run_pipeline                  # draw a route, run everything
python -m pose_alignment.run_pipeline --roi outputs/alignment/routes/my_route.json    # reuse a drawn route
python -m pose_alignment.run_pipeline --roi-dir outputs/alignment/mapillary_roi_19   # resume where it stopped
python -m pose_alignment.run_pipeline --roi-dir outputs/alignment/mapillary_roi_19 --from reconstruct --one-leg
python -m pose_alignment.run_pipeline --roi-dir outputs/alignment/mapillary_roi_19 --from reconstruct --tag test
```

Every stage script also runs on its own (`python -m pose_alignment.<script> --help`;
the first two as `python -m data_ingest.mapillary_sequences` and
`python -m segmentation.street_masks`).

- **`--from` / `--to`**: first and last stage. The default starts a new ROI at the GUI, or
    an existing one (`--roi-dir`) at its first unfinished stage.
- **`--one-leg`**: the whole route as one reconstruction (no splits at turns or by length).
    Better on long straight streets: on roi_19 (650 m) four separate legs each fitted the
    walls but landed 19–45 m apart at their junctions; one leg aligned all 84 frames at
    1.00 m.
- **`--max-frames`** (default 60), **`--max-turn`** (35°): where legs are split otherwise.
- **Linked legs.** Legs are reconstructed one by one, but tied together in two ways:
    - **Overlap (`--leg-overlap`, default 8 frames):** each leg runs on into the next leg's first 8
        frames, including the frames through a turn of up to 30 frames. Two consecutive legs then
        share those frames.
    - **One focal length per camera (`--no-shared-focal` turns it off):** after the first pass, each
        camera's focal is set to the median of its legs' refined focals, weighted by frames. Every leg
        is then solved again on its own features and matches, with the focal held there and only the
        distortion refined. This adds one mapper run per leg, not new feature matching. Solved alone,
        a leg's focal drifts and stretches it along the street.
    - **Joint placement (align; `--no-link` turns it off):** each leg is first fitted to OSM on its
        own. The fits of linked legs are then refined together, so the cameras they share land in the
        same place from both legs. That lines the legs up end to end and carries their scale across.
        A leg whose walls leave it free to slide along the street is held by its neighbours.
- **Rigs (default; `--solo-cameras` turns it off):** the cameras of a multi-camera rig are
    solved together. This is always on for a 360 camera's left and right views, which form an
    exact rig.
    - **Which cameras:** the siblings that take their frames at the same moments as the primary
        camera (within 20 ms, for at least 30% of their frames).
    - **What it does:** each rig frame gets one pose, and each camera a fixed pose in the rig, so the
        forward and rear cameras keep the side cameras from drifting into curves.
    - **Results on roi_11** (2017 Roush, five cameras): the side cameras' bow went from 2–40 m to
        0.1 m, and reliable frames rose from 168 to 238 of about 500.
    - **Mounting angles:** start from Mapillary's rotations and are solved in a first pass. The
        second pass holds them, with every camera at the reference camera's position. A side camera
        shares no points with the others, so its offset in the rig can't be determined, and a free
        offset slid it and its facades 2–12 m.
    - **Placement on the map:** align places the rig with one transform, fitted with all the
        cameras' building points. A camera facing buildings missing from OSM (roi_11's right side)
        then still lands where the rest of the rig puts it.
    - **Cameras out of sync:** cameras that fire at other moments (roi_6's motorcycle cameras, 200 ms
        apart in two of its legs) are solved alone, as before.
- **Speed:**
    - Keypoints for every leg camera are extracted and matched in one GPU process, each frame
        once. Before, each leg camera started its own process, and overlap frames were extracted twice.
    - The shared-focal pass skips a leg camera whose own focal is already within 1% of the shared
        value.
    - **Matching look-ahead and RANSAC tries.** Each frame is matched with its next 6 frames, with
        RANSAC capped at 2000 tries per pair in COLMAP's match checking, when frames are close
        together (nearest-neighbour matching, under 6 m apart). Frames further apart (LightGlue) keep
        10 frames and 10,000 tries. `--match-overlap` and `--ransac-trials` override both.
        - roi_13: about 1.5 min faster with the same results.
        - roi_19, frames 7.6 m apart: the shorter look-ahead made every leg 0.2–0.45 m worse, so
            widely spaced frames keep the longer one.
    - Segmentation runs 4 frames per model call: 12% faster, same masks.
- **`--gap-factor`** (default 3): side and rear cameras keep only their longest run of frames
    with no jump bigger than 3× their usual spacing (and at least 6 m). Across a bigger jump they
    share almost no view, and a few false matches can join the two sides in the wrong place (the
    rear camera of roi_11 leg01). Forward cameras are never cut. `0` turns this off.
- **`--split-weak`**: drop weakly anchored frames and fit each side of the gap separately
    (off by default; see the reconstruct stage).
- **`--matcher`**: `auto` (default) picks nearest-neighbour matching for frames under 6 m
    apart and LightGlue otherwise; `nn` or `lightglue` forces one.
- **`--tag t`**: a separate run of reconstruct and align (`data/reconstruction_t`,
    `data/alignment_t`, `results/t`); the calibration is shared.
- The mapillary stage always writes a new `outputs/alignment/mapillary_roi_N` folder; `--view right`
    prefers cameras looking right, `--sequence ID` forces a sequence, `--no-rig` skips the
    rig's other cameras.

Every stage script also runs on its own (`python <script> --help` lists its options), e.g.
`python -m pose_alignment.reconstruct outputs/alignment/mapillary_roi_19 --max-turn 180 --trim-deg 180 --max-frames 200`.

## 1. Draw the route (`route_gui.py`)

Opens a map in your browser. Click along the streets you want; turns are fine. **New
street** starts a separate line. The corridor half-width must cover your clicking error but
not reach parallel streets (15 m works for most roads). **Use this route** saves it.

## 2. Fetch the imagery (`data_ingest/mapillary_sequences.py`)

On its own, Mapillary is searched (`common.providers.MapillaryClient`) in small tiles along
each line (a large query returns only a sample); in the main pipeline the fetch stage's
records are used.
Images outside the corridor are dropped. Each sequence (one capture session of one camera)
is split into continuous runs, and the best run is picked:

1. enough frames along its longest straight stretch;
2. frames dense enough to reconstruct: median spacing at most 10 m (`--max-spacing`);
3. the largest share of the drawn line (in 25% steps);
4. which way the camera looks: a 360 run first (its right and left views both face the
    facades), then a camera facing the side of the street, then one facing forward or back
    (it sees the facades at a grazing angle);
5. the densest spacing, then the most frames.

**360° sequences.** A 360 run wins over perspective runs that cover the line as well as
it does. `--pano prefer` takes a 360 run whenever it has enough dense frames, even over a
perspective run covering more; `--pano off` ignores 360 runs.

A picked 360 run is downloaded at full resolution. Each frame is cut into two square pinhole
views, one facing right of travel and one facing left (`--pano-fov`, default 90°, at most
2048 px). The two views are saved as a two-camera rig:

- `line<L>_cam0_<id>`: the right view, the primary;
- `line<L>_cam1_<id>`: the left view.

The views point in a fixed direction in the 360 image, ±90° from travel by the run's median
compass. Each view's `cameras.json` carries a `pano` header (side, yaw, field of view, source
size). Its compass and rotation are turned to the view's direction.

Every later stage handles the views like any rig camera, with one difference: the focal length
is exact and there is no distortion. Calibrate takes that focal, and the gravity from the
view's turned rotation. Reconstruct keeps the camera fixed, so the focal cannot drift and
stretch the model along the street.

Rig siblings (vehicles with several cameras upload one sequence per camera) are added.
Frames are saved as `data/mapillary/line<L>_cam<k>_<id>/frame_0001.jpg …` with
`cameras.json` (Mapillary id, time, raw GPS, Mapillary's computed position and rotation,
compass, make, model, calibration). About 10 s per 100 frames.

## 3. Segment (`segmentation/street_masks.py`)

Mask2Former (Swin-Large, Mapillary Vistas; `segmentation/model.py`, shared with texturing's
obstruction removal) labels every pixel, reduced to four values per mask: 0 other, 100 road,
200 building, 255 sky and moving things. The class maps themselves are kept too
(`data/labels/<image key>.png`) for texturing. About 0.3 s per frame on
an Apple-silicon GPU. `--previews` saves colour overlays.

## 4. Calibrate (`calibrate.py`)

Everything comes from Mapillary's own structure from motion, in each frame's `cameras.json`:

- **Calibration:** the sequence's most common `camera_parameters` (focal as a fraction of the
    long side, k1, k2). Mapillary's focal is sometimes far off (roi_15's iPhone reports 1.338
    against about 0.8; a Google Pixel 3 downtown got 0.976, 0.918 and 0.732), so each sequence is
    checked against the ROI's other sequences of the same camera (make, model and image size).
    When at least two others are calibrated and this one's focal is more than 15% from their
    median, it takes the calibration of the one closest to that median (with only two, both are
    kept and a warning is printed when they differ by more than 15%; `source:   mapillary_median`, with a `correction` saying why). Without any Mapillary calibration a
    sequence starts from COLMAP's usual 1.2 x the long side with no distortion (`source:   default`), with a warning. Bundle adjustment refines the focal and distortion either way.
- **Gravity:** each frame's up direction in the camera frame from its `computed_rotation`
    (world-to-camera in east/north/up): the rotation applied to the world's up. Frames without
    one are interpolated between their neighbours. A sequence without any rotation gets no
    gravity: reconstruct solves it without gravity priors, and align levels it from the plane
    of its road points (or, with too few, from the cameras' own down axes), with a warning.

It takes no time to compute and needs no model. Calibrations from the earlier GeoCalib
version of this stage are recalculated when found (`--force` recalculates every sequence).

**The GeoCalib version.** Until this change the stage estimated the focal, distortion and
gravity from the photos with GeoCalib (ECCV 2024). Its measurements, kept for reference:
GeoCalib's per-frame gravity was 0.44-0.48° from the COLMAP models' rotations on cars and 1.27°
on a motorcycle (roi_6, 19, 24, 27; those models were built with GeoCalib's own gravity
priors); its focal was about 5.4% off COLMAP's refined one (12 normal-lens sequences); it took
about 0.8 s per frame; and lenses wider than its model (roi_21's 2017 Roush rig: 0.42-0.65 for
six identical cameras, Mapillary 0.48) needed Mapillary's calibration anyway. Starting
smartphones from Mapillary's focal changed nothing where frames are close (roi_2, 4, 5, 7, 8,
9). On 4 frames compared after the change, GeoCalib's and Mapillary's gravity differed by
2.6-3.8°, about the same on every frame.

To see how consistent Mapillary's gravity is with a reconstruction, run `gravity_check.py`
(below); the alignment's `summary.json` and `shape_check.py` show the result on the fit.

## 5. Reconstruct (`reconstruct.py`)

**Legs.** Each line's capture is split at turns over 35° (frames in the corner trimmed) and
into chunks of at most 60 frames; rig cameras get the frames in each leg's time window.

Per leg and camera:

1. **Feature masks:** burned-in overlays (timestamps, bonnets) and the segmentation's sky,
    vehicles and people.
2. **Camera:** one per sequence, COLMAP's `RADIAL` from the calibration's focal, k1 and k2.
    Bundle adjustment refines it. (The GeoCalib version switched strongly distorted lenses,
    GeoCalib k1 below −0.2, to `RADIAL_FISHEYE`.)
3. **Features:** ALIKED keypoints, each frame matched with its next 10:
    - frames under 6 m apart: mutual nearest neighbour of the descriptors (`nn_features.py`),
        2–6 s per leg;
    - 6 m and over: LightGlue (`learned_features.py`), about 15× slower. With
        nearest-neighbour, sequences 8–10 m apart (roi_4, 5, 9, 19) reconstructed unstably
        (bent, crushed or lost frames).
4. **Gravity priors:** the calibration's down direction per frame (Mapillary's rotations)
    goes into COLMAP's pose priors (no positions: GPS is not used). A sequence without
    rotations has none, and the mapper runs without gravity for it.
5. **COLMAP 4.2 `global_mapper`** (GLOMAP) with gravity-aided rotation averaging, then
    `point_triangulator` re-triangulates every match with the poses fixed (more road and
    building points).
    - **Fallback:** if any camera lands behind an earlier one along the leg, the leg is solved
        again with COLMAP's incremental mapper on the same matches, and kept if it holds capture
        order. GLOMAP sets all rotations at once from frame-pair rotations; where a few pairs agree
        on a wrong rotation (a parked truck filling the view on roi_11, a skipped frame on roi_20)
        it turns those cameras and slides a stretch back over the previous one. No seed or outlier
        setting avoided it, while the incremental mapper kept both in order. Elsewhere GLOMAP is
        as good or better (roi_10, roi_5), so it stays first. `--mapper incremental` forces it.
6. **Weak frames (off by default, `--split-weak`):** frames left with under 0.35× the leg's
    median number of 3D points (e.g. a parked truck filling the view) are dropped and the leg
    solved again, and the alignment fits each side of the gap separately. It fixed roi_11's
    truck leg, but on roi_20 the two sides ended 72 m apart where Mapillary has 41 m, so it
    isn't trusted by default yet.

## 6. Align to OSM (`align.py`)

Per leg and camera:

1. **Level** the model with the calibration's gravity; without it, from the road points'
    plane, or the cameras' own down axes (`summary.json`: `gravity.levelling`).
2. **Ground:** median height of the road points near the path.
3. **Building points:** labelled Building, seen in 3+ frames, 2.5–40 m up, thinned to one
    per 0.5 m.
4. **Fit:** one 2D similarity (heading, scale, position) placing the building points as close
    as possible to the OSM footprint edges: a global search around Mapillary's positions,
    then a robust refinement. Every building point counts, including side walls, corners and
    gaps, which is what pins the position along the street. GPS gives the starting placement
    and weak priors only.
5. **Pieces (only with `--split-weak`):** where weak frames were dropped, each side of the
    gap gets its own fit; the output is still one leg.
6. **Reliable** when the building points end within 2 m (median, `--max-error`) of the
    footprints, at most 20% of the cameras end inside buildings, and no camera lies behind an
    earlier one along the leg (a folded reconstruction can still fit its walls).

Outputs:

| Path                               | Contents                                                             |
| ---------------------------------- | -------------------------------------------------------------------- |
| `data/alignment/<sequence>.json`   | aligned cameras: lon, lat, height, heading, pitch, roll, reliable    |
| `data/alignment/<leg>_<sequence>/` | `stats.json`, `transform.json`, `plan.png` (GPS placement vs fitted) |
| `data/alignment/summary.json`      | frames aligned and reliable, per-leg fit                             |
| `results/overview.png`             | all aligned cameras on the map (blue reliable, red unreliable)       |
| `results/fit_<leg>_<sequence>.png` | the leg's point cloud on the footprints, plan and 3D                 |

Fitting the whole point cloud replaced fitting extracted facade lines to street-facing
walls: against a three-pass reference on roi_17, single passes came out 1.9–7.7 m away
instead of 9–14 m, and failures are reported as unreliable instead of passing.

## Checking

- `python -m pose_alignment.gravity_check <roi> [--tag t]`: per leg and sequence, how consistent
    Mapillary's per-frame gravity is with the reconstructed cameras. Every camera's "down",
    turned into the model by its COLMAP rotation, should point the same way (a street has one
    up); the spread around their robust mean is reported. The reconstruction used that gravity
    for its priors, so this is a lower bound, and a bias the same on every frame only shows where
    the camera turns. Writes `results/[tag/]gravity_check.json`.
- `python -m pose_alignment.shape_check <roi> [--tag t]`: per leg, how far the camera path bends compared
    with Mapillary's track, how well each camera's tilt agrees with the calibration's gravity, and
    the scale drift from the start to the end of the leg. Use it alongside the fit: a bent or
    crushed reconstruction can still fit its walls.

## Results so far

| ROI              | Camera                                          | Frames reliable | Building points to footprints                                      |
| ---------------- | ----------------------------------------------- | --------------- | ------------------------------------------------------------------ |
| roi_2            | iPhone                                          | 28/30           | 0.94 m                                                             |
| roi_4            | iPhone, 8.6 m apart                             | 30/30           | 1.14 m                                                             |
| roi_5            | iPhone, 9.9 m apart                             | 43/48           | 1.20 m                                                             |
| roi_7            | iPhone, 2 legs                                  | 36/37           | 0.71 / 1.05 m                                                      |
| roi_8            | iPhone                                          | 22/23           | 0.58 m                                                             |
| roi_11           | motorcycle rig, 2 cameras                       | 79/131          | 0.85–0.94 m; one leg at a construction site that doesn't match OSM |
| roi_19           | iPhone, 650 m, one leg                          | 84/84           | 1.00 m                                                             |
| main_street_east | motorcycle                                      | 33/33           | 0.74 m                                                             |
| roi_9            | iPhone, forward, 10 m apart, buildings far back | 0/33            | 3.2 m                                                              |

### Example fits

Images of example fits (roi_19, roi_5, roi_11 and roi_20) are on the
[`NickNack08-patch-1` branch](https://github.com/Qualcomm-MDP/PSM/tree/NickNack08-patch-1/Alignment_Pipeline/example_results).
Each shows one camera's reconstruction after alignment: a plan view of the building points
(red), other points (blue) and road points (grey) over the OSM footprints with the camera
path in black, and the same in 3D against the footprints extruded to their OSM heights.

Typical time for 80–100 frames, measured with the GeoCalib version: about 5 minutes
(GeoCalib 1 min, which calibration no longer takes; LightGlue matching 2.5 min when used,
COLMAP 0.5 min, alignment 0.3 min).

## Limits

- A fit per leg cannot undo drift within the leg (a reconstruction that shrinks or bends
    along the street); measured drift on roi_19's 650 m leg was about 7%.
- Forward-facing cameras with widely spaced frames and buildings set far back (roi_9)
    reconstruct poorly.
- Scenes that don't match OSM (construction sites, fences, open frames) stay unreliable.
- Several captures of the same street are not combined.

## Setup

1. The project's environment (`poetry install`). It includes LightGlue (for ALIKED
    features), installed from its GitHub repository, so `git` must be available.
    First runs download the Mask2Former (about 900 MB), ALIKED and LightGlue weights.
2. COLMAP 4.2 or newer on the PATH (`brew install colmap` on macOS), or `COLMAP_BIN` in
    `.env` set to its executable.
3. A Mapillary client access token from https://www.mapillary.com/dashboard/developers,
    set as `MAPILLARY_TOKEN` (or the main pipeline's `MAPILLARY_ACCESS_TOKEN`). Do not
    commit it.

OSM buildings come from the public Overpass API (`common.providers.OverpassClient`; the main OSM API when Overpass is down)
and are saved per ROI. No keys are needed for OSM or the GUI's basemaps.

## Data layout

```
outputs/alignment/
  outputs/alignment/routes/                             drawn routes (route_gui.py)
  mapillary_roi_N/
    data/
      mapillary/    roi.json  sequences.json  line1_cam0_<id>/{frame_0001.jpg ..., cameras.json}
      masks/        classes.json  line1_cam0_<id>/frame_0001.png ...
      calibration/  line1_cam0_<id>.json
      reconstruction/  legs.json  leg01/line1_cam0_<id>/{sparse/0/txt, colmap.log, ...}
      alignment/    line1_cam0_<id>.json  summary.json  osm_buildings.json  leg01_line1_cam0_<id>/
      reconstruction_<tag>/  alignment_<tag>/      tagged runs
    results/
      overview.png  fit_leg01_line1_cam0_<id>.png  <tag>/
```
