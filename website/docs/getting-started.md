# Getting Started

There are three ways to run POSM. Pick one:

| Way                                                 | Best for               | You need                                                                             |
| --------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------ |
| [Web app in Docker](#web-app-in-docker-recommended) | Most people            | [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine) |
| [Command line in Docker](#command-line-in-docker)   | Scripts and batch runs | Docker                                                                               |
| [From source](#from-source-poetry)                  | Developing POSM        | Python 3.11–3.13, [Poetry](https://python-poetry.org/docs/), Git                     |

All of them need a **Mapillary access token** (free).

:::tip Getting a Mapillary token

Sign in at [mapillary.com](https://www.mapillary.com/), open your dashboard, go to **Developers**, register an app, and copy its **client token** (it starts with `MLY|`).

:::

## Web app in Docker (recommended)

```bash
git clone https://github.com/Qualcomm-MDP/Photorealistic-Street-Maps.git
cd Photorealistic-Street-Maps
docker compose up --build api
```

Open <http://localhost:8000>. On first use:

1. Click the **gear icon** (Settings), paste your Mapillary token, and press **Save token**. If no `.env` file exists yet, POSM creates one with the token.
2. Pick an area: search for an address, or press **Draw area** and drag a box on the map.
3. Press **Generate model**. The timeline and the pipeline log follow the job live, and the model appears in the 3D preview when it's done.

See [Web App](web-app.md) for everything the UI can do.

The image runs natively on Intel/AMD and ARM machines, including Apple Silicon Macs. Model weights download on the first job and are kept in a Docker volume.

:::note NVIDIA GPU

On Linux (or Windows with WSL2) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), use the GPU image instead:

```bash
docker compose --profile gpu up --build api-gpu
```

Docker on macOS can't use the GPU; CPU runs work, just more slowly.

:::

## Command line in Docker

```bash
docker compose build pipeline
docker compose run --rm pipeline --bbox -83.71978 42.29025 -83.71205 42.29422 --output output/map.glb
```

The model is written to `./output/map.glb`. The token comes from your env file (see [Configuration](configuration.md)); with `APP_ENV=development` it reads `.env.development`:

```bash
APP_ENV=development docker compose run --rm pipeline --bbox W S E N --output output/map.glb
```

## From source (Poetry)

```bash
git clone https://github.com/Qualcomm-MDP/Photorealistic-Street-Maps.git
cd Photorealistic-Street-Maps
poetry install
```

Create a `.env` file with your token, either with the interactive builder or by hand:

```bash
poetry run python dotenv_builder.py
```

```bash
# .env
MAPILLARY_ACCESS_TOKEN="MLY|..."
```

Then run the command line:

```bash
poetry run python src/main.py --bbox -83.71978 42.29025 -83.71205 42.29422 --output map.glb
```

or the web app:

```bash
poetry run uvicorn api.app:create_app --factory --app-dir src
```

:::note PyTorch from source

On Linux, `poetry install` gets PyTorch from PyPI, which includes CUDA (several GB) even without an NVIDIA GPU. The Docker images install the right build for you (CPU or CUDA).

:::

## Command-line options

Without `--bbox` the CLI asks for the coordinates; without `--output` it opens a save dialog.

| Option                  | Description                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--bbox W S E N`        | Area in degrees: west, south, east, north.                                                                                                        |
| `--output PATH`         | Where to write the `.glb`.                                                                                                                        |
| `--no-seg`              | Skip obstruction removal (Mask2Former + LaMa). Much faster; facade textures keep cars, people and poles.                                          |
| `--terrain`             | Put buildings and cameras on real ground heights and add the ground to the `.glb`. See [Terrain & Aerial Photos](pipeline/terrain-and-aerial.md). |
| `--naip`                | Texture the ground and roofs with NAIP aerial photos (continental US only). Turns on `--terrain`.                                                 |
| `--min-quality SCORE`   | Skip Mapillary images whose `quality_score` (0–1) is below `SCORE`. Images without a score are kept.                                              |
| `--profile FILENAME`    | Save per-stage timing and memory to a JSON file. See [Evaluation](pipeline/evaluation.md).                                                        |
| `--dump-mapillary PATH` | Write the raw Mapillary metadata to a JSON file, for debugging.                                                                                   |

## Finding bounding box coordinates

In the web app you pick the area on the map. For the command line:

1. Go to [OpenStreetMap](https://www.openstreetmap.org/) and navigate to your area.
2. Click **Export** in the top menu, then **Manually select a different area**.
3. Read the four values (west, south, east, north).

:::warning Area size

Bigger areas mean more buildings and many more street photos, so runs get long. Start with a few city blocks. The web app limits each side to 2.5 km.

:::

## Viewing the output

The `.glb` opens in:

- the **POSM web app** (3D preview, with a larger view)
- **[Blender](https://www.blender.org/)** (File → Import → glTF 2.0)
- **[glTF Viewer](https://gltf-viewer.donmccurdy.com/)** (web, drag and drop)
- **Windows 3D Viewer** and **macOS Quick Look**

## Project structure

```
Photorealistic-Street-Maps/
├── src/
│   ├── main.py                  # Command-line entry point
│   ├── pipeline.py              # The pipeline: stages, export (shared by CLI and web)
│   ├── api/                     # Web app backend (FastAPI): jobs, logs, settings, search
│   ├── data_ingest/ingest.py    # Fetch stage: OSM + Mapillary
│   ├── terrain/terrain.py       # Terrain stage: ground heights
│   ├── mesh_builder/extrude.py  # Build-mesh stage: footprints → 3D buildings
│   ├── naip/naip.py             # Aerial-photo stage: NAIP ground and roofs
│   ├── texturing/               # Texture stage: camera projection and texture atlas
│   ├── segmentation/            # Mask2Former + LaMa obstruction removal
│   └── common/                  # Pipeline framework, reporter, providers, geometry helpers
├── web/                         # Web app frontend (HTML, CSS, JavaScript)
├── website/                     # This documentation site (Docusaurus)
├── tests/                       # Pytest suite
├── Dockerfile                   # CPU and CUDA images
├── docker-compose.yml           # pipeline / api services, CPU and GPU
└── dotenv_builder.py            # Interactive .env setup
```
