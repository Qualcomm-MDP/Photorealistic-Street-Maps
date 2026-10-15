[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Qualcomm-MDP/Photorealistic-Street-Maps/main.svg)](https://results.pre-commit.ci/latest/github/Qualcomm-MDP/Photorealistic-Street-Maps/main)

# Photorealistic Street Maps

This project is a proposed pipeline to create photorealistic 3D cityscape.

# Quickstart

TODO: Write about install and running the pipeline towards the end

## Running with Docker

Docker is the easiest way to run the pipeline on any OS without installing Python, Poetry, or PyTorch yourself.

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

Docker on macOS cannot use the GPU. CPU runs work, but obstruction removal takes a few seconds per image; pass `--no-seg` to skip it.

Optional environment variables (set in `.env` or with `-e`):

| Variable     | Default                                               | Purpose                                                                                                                        |
| ------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `SEG_DEVICE` | auto (`cuda` if available, else `cpu`)                | Force the segmentation device (`cuda`, `mps`, `cpu`).                                                                          |
| `SEG_MODEL`  | `facebook/mask2former-swin-large-cityscapes-semantic` | Any Mask2Former Cityscapes semantic checkpoint, e.g. `facebook/mask2former-swin-tiny-cityscapes-semantic` for faster CPU runs. |

# Contribution

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
