[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Qualcomm-MDP/PSM/main.svg)](https://results.pre-commit.ci/latest/github/Qualcomm-MDP/PSM/main)

# Photorealistic Street Maps

This project is a proposed pipeline to create photorealistic 3D cityscape.

# Quickstart

## Running locally (full pipeline, with camera pose alignment)

This runs everything, including the camera pose alignment (`src/pose_alignment/`), which
reconstructs the street photos with COLMAP and fits them to the OpenStreetMap buildings
before texturing. Until it is merged into `main`, it lives on the
`feat/colmap-pose-alignment` branch.

**You need:** Python 3.11-3.14, [Poetry](https://python-poetry.org/), git, a Mapillary
client access token ([developer dashboard](https://www.mapillary.com/dashboard/developers)),
and, for aligned poses, [COLMAP](https://colmap.github.io/install.html) 4.2 or newer.

1. Get the branch and install the Python packages (GeoCalib and LightGlue are installed from
    their GitHub repositories, so git must be available):

    ```
    git clone https://github.com/Qualcomm-MDP/PSM.git
    cd PSM
    git checkout feat/colmap-pose-alignment
    pip install poetry
    poetry install
    ```

    The first run also downloads model weights (Mask2Former, about 900 MB; LaMa; GeoCalib;
    ALIKED and LightGlue) into your Hugging Face and PyTorch caches.

2. Install COLMAP 4.2+:

    - macOS: `brew install colmap`
    - Linux: build it from source ([instructions](https://colmap.github.io/install.html));
        distribution packages are often older than 4.2.
    - Windows: the release binaries from [GitHub](https://github.com/colmap/colmap/releases).

    If `colmap` is not on your `PATH`, set `COLMAP_BIN` in `.env` to its executable. Without
    COLMAP everything else still runs, and aligned poses fall back to Mapillary's computed
    ones (the log says so).

3. Put your Mapillary token in `.env` (see `.env.example`, or run `python dotenv_builder.py`):

    ```
    MAPILLARY_ACCESS_TOKEN="..."
    ```

    In the web app you can also add it under Settings.

4. Run the web app, then open [http://localhost:8000](http://localhost:8000):

    ```
    poetry run uvicorn api.app:create_app --factory --app-dir src
    ```

    Draw a box on the map and generate. The Options panel has **Camera poses** (Aligned to
    OpenStreetMap by default, or Mapillary's computed or raw GPS poses), **Aligned photos
    only**, **Remove obstructions** and **Fill obstructions** (off: obstructions are found
    but not painted over, and each wall takes the photos they hide least), **Terrain** and
    **Aerial photos**. The job's log shows every stage as it runs, and the 3D preview's
    **Cameras** and **Point cloud** buttons show the photos' camera positions (green
    aligned, orange Mapillary computed, red raw GPS) and the alignment's reconstructed points.

    Or from the command line (`--bbox WEST SOUTH EAST NORTH`):

    ```
    poetry run python src/main.py --bbox -77.61355 43.15326 -77.61140 43.15684 --poses colmap --output output/model.glb
    ```

    | Option                                              | Effect                                                                                 |
    | --------------------------------------------------- | -------------------------------------------------------------------------------------- |
    | `--poses raw` / `computed` / `colmap`               | Mapillary's raw GPS (the CLI default), Mapillary's computed poses, or aligned to OSM   |
    | `--aligned-only`                                    | texture only from photos the alignment placed                                          |
    | `--no-fill`                                         | find obstructions but do not paint them over                                           |
    | `--no-seg`                                          | no obstruction removal at all                                                          |
    | `--texture best`                                    | each face from its single best photo, instead of blending every photo that sees a wall |
    | `--terrain`, `--naip`                               | real ground heights; NAIP aerial photos on the ground and roofs                        |
    | `--alignment-dir outputs/alignment/mapillary_roi_N` | reuse an earlier alignment instead of running it again                                 |

**What a run does.** It fetches the OSM buildings and the Mapillary photos around the box,
then picks the captures to use: along each OSM street, the densest run that covers it (a
360 camera first, as its right and left views both face the facades, then cameras facing
the side of the street), and further runs where that one ends. With aligned poses, those
photos are downloaded, segmented, calibrated (GeoCalib), reconstructed (COLMAP) and fitted
to the OSM building outlines, in `outputs/alignment/mapillary_roi_N/` (figures in its
`results/`). The buildings are extruded from OSM, every wall is textured by blending all
the photos that see it, and the model is saved as a `.glb`, with `<model>.overlays.json`
(cameras and point cloud) next to it.

**Time and hardware.** A few blocks (300-450 frames) take about 10-15 minutes on an Apple
silicon laptop, most of it the alignment. The alignment's segmentation, GeoCalib and
feature matching use the Apple GPU (or an NVIDIA GPU, with CUDA); COLMAP runs on the CPU
with Homebrew's build, and texturing's obstruction removal uses an NVIDIA GPU if there is
one, else the CPU.

More on the alignment: [src/pose_alignment/README.md](src/pose_alignment/README.md).

## Running with Docker

Docker is the easiest way to run the pipeline on any OS without installing Python, Poetry, or PyTorch yourself.

The image does not include COLMAP yet, so in Docker the camera pose alignment does not run:
`--poses colmap` (and the web app's "Aligned to OpenStreetMap") fall back to Mapillary's
computed poses. Everything else (capture choice, 360 views, segmentation, blended textures,
overlays) runs the same. For aligned poses, run locally (above). The web app in Docker:
`docker compose up api`, then open [http://localhost:8000](http://localhost:8000).

1. Create a `.env` file with your Mapillary token (see `.env.example`, or run `python dotenv_builder.py`):

    ```
    MAPILLARY_ACCESS_TOKEN="..."
    ```

    To use `.env.development` or `.env.production` instead, set `APP_ENV` when running, e.g. `APP_ENV=development docker compose run ...`. Env files are read when the container starts and are never copied into the image.

2. Build and run (CPU, works on Linux, macOS, and Windows):

    ```
    docker compose build pipeline
    docker compose run --rm pipeline --bbox -83.71978 42.29025 -83.71205 42.29422 --output output/map.glb
    ```

    The mesh is written to `./output/map.glb`. Model weights are downloaded on the first run and cached in the `model-cache` Docker volume.

3. With an NVIDIA GPU (Linux, or Windows with WSL2, plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

    ```
    docker compose --profile gpu build pipeline-gpu
    docker compose --profile gpu run --rm pipeline-gpu --bbox -83.71978 42.29025 -83.71205 42.29422 --output output/map.glb
    ```

Docker on macOS cannot use the GPU. CPU runs work, but obstruction removal takes a few seconds per image; pass `--no-seg` to skip it. The CPU image is built for your machine's own architecture, so it runs natively on Apple Silicon (M1/M2/M3) as well as on Intel/AMD.

Optional environment variables (set in `.env` or with `-e`):

| Variable     | Default                                                     | Purpose                                                                                                                                                      |
| ------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SEG_DEVICE` | auto (`cuda` if available, else `cpu`)                      | Force the segmentation device (`cuda`, `mps`, `cpu`).                                                                                                        |
| `SEG_MODEL`  | `facebook/mask2former-swin-large-mapillary-vistas-semantic` | Any Mask2Former semantic checkpoint (Mapillary Vistas or Cityscapes classes), e.g. `facebook/mask2former-swin-tiny-cityscapes-semantic` for faster CPU runs. |

# Contribution

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
