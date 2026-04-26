# Getting Started

This guide walks you through installing and running the POSM pipeline on your machine.

## Prerequisites

- **Python 3.11 -- 3.13** (3.14.1 is excluded due to a known issue)
- **Poetry** for dependency management ([install guide](https://python-poetry.org/docs/))
- **Git** for cloning the repository
- **A Mapillary API token** (free, required for street-view image access)

!!! note "GPU Support"

    The pipeline runs on CPU by default. If you have an NVIDIA GPU with CUDA support, segmentation and inpainting models will automatically use it for significantly faster processing.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Qualcomm-MDP/Photorealistic-Street-Maps.git
cd Photorealistic-Street-Maps
```

### 2. Verify Poetry is installed

```bash
poetry --version
# Expected output: Poetry (version x.x.x)
```

If Poetry is not installed, follow the [official installation guide](https://python-poetry.org/docs/).

### 3. Install dependencies

```bash
poetry install
```

This installs all runtime and development dependencies, including PyTorch, Trimesh, OpenCV, Transformers, and more.

### 4. Configure environment variables

The pipeline requires a Mapillary API token. Run the interactive `.env` builder:

```bash
poetry run python dotenv_builder.py
```

You will be prompted to select an environment (`development`, `production`, or `none`) and provide values for each required variable. At minimum, set your `MAPILLARY_ACCESS_TOKEN`.

Alternatively, create a `.env` file manually:

```dotenv
MAPILLARY_ACCESS_TOKEN="your_token_here"
```

!!! tip "Getting a Mapillary Token"

    Sign up at [mapillary.com](https://www.mapillary.com/) and generate an access token from your dashboard under **Developers > API**.

## Running the Pipeline

Launch the pipeline with:

```bash
poetry run python src/main.py
```

You will be prompted to enter the bounding box coordinates for your region of interest:

```
WEST (Minimum Longitude): -83.71978
SOUTH (Minimum Latitude): 42.29025
EAST (Maximum Longitude): -83.71205
NORTH (Maximum Latitude): 42.29422
```

The pipeline will then:

1. **Fetch** building footprints from OSM and street-view images from Mapillary
2. **Build** 3D meshes by extruding OSM building geometries
3. **Texture** the meshes using camera-projected street-view imagery
4. **Export** the result as a `.glb` file (a save dialog will appear)

### Command-Line Options

```bash
# Run with performance profiling (saves metrics to a JSON file)
poetry run python src/main.py --profile performance.json

# Skip obstruction removal (faster, but textures may include trees/cars)
poetry run python src/main.py --no-seg
```

| Flag                 | Description                                                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--profile FILENAME` | Enable the built-in profiler and save stage-level metrics (wall time, CPU time, memory, GPU VRAM) to the specified JSON file.                                  |
| `--no-seg`           | Disable the Mask2Former + LaMa obstruction removal pipeline. Produces faster results but street-view textures will retain trees, cars, and other obstructions. |

## Finding Bounding Box Coordinates

The easiest way to get bounding box coordinates for a region:

1. Go to [OpenStreetMap](https://www.openstreetmap.org/)
2. Navigate to your area of interest
3. Click **Export** in the top menu
4. The bounding box values (min lat, min lon, max lat, max lon) are displayed

!!! warning "Region Size"

    Larger regions produce more buildings and require more Mapillary images, increasing runtime. Start with a small area (a few city blocks) to test. The target benchmark is under 25 minutes for a 247-acre region.

## Viewing the Output

The pipeline exports a `.glb` file that can be viewed in:

- **[Blender](https://www.blender.org/)** (free, full-featured 3D editor)
- **[glTF Viewer](https://gltf-viewer.donmccurdy.com/)** (web-based, drag and drop)
- **Windows 3D Viewer** (built into Windows 10/11)
- **macOS Quick Look** (spacebar preview in Finder)

## Project Structure

```
Photorealistic-Street-Maps/
├── src/
│   ├── main.py                  # Entry point & pipeline orchestration
│   ├── data_ingest/
│   │   └── ingest.py            # OSM + Mapillary data fetching
│   ├── mesh_builder/
│   │   └── extrude.py           # 2D footprint → 3D mesh extrusion
│   ├── texturing/
│   │   └── tex_projection.py    # Camera projection & texture atlas
│   ├── segmentation/
│   │   └── obstruction.py       # Mask2Former + LaMa inpainting
│   └── common/
│       ├── pipeline_chain.py    # Pipeline framework
│       ├── profiler.py          # Performance profiler
│       ├── constants.py         # BoundingBox, API URLs
│       ├── MeshUtils.py         # Geometry helpers
│       ├── MeshExport.py        # GLB export
│       ├── ProgressMonitor.py   # CLI progress display
│       ├── conversions/         # Coordinate conversion utilities
│       └── providers/           # OSM, Mapillary, Overpass clients
├── tests/                       # Pytest test suite
├── docs/                        # This documentation (MkDocs)
├── mkdocs.yml                   # MkDocs configuration
├── pyproject.toml               # Poetry project config
└── dotenv_builder.py            # Interactive .env setup
```
