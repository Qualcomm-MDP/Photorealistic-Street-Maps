# Configuration

POSM is configured with environment variables, usually from an env file. The web app can also save the Mapillary token and job defaults itself (see [Web App](web-app.md#settings)).

## Env files

`APP_ENV` picks which file is read, both by the app and by `docker-compose.yml`:

| `APP_ENV`          | File               |
| ------------------ | ------------------ |
| not set (or empty) | `.env`             |
| `development`      | `.env.development` |
| `production`       | `.env.production`  |
| any other `name`   | `.env.name`        |

```bash
APP_ENV=development docker compose up api
```

Env files are read when the program or container starts; Docker never copies them into the image. Create one with `poetry run python dotenv_builder.py`, by hand, or by saving a token in the web app's Settings (which creates the file if it doesn't exist). See `.env.example` for the format.

## Variables

| Variable                 | Default                                               | Purpose                                                                                                                                          |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MAPILLARY_ACCESS_TOKEN` | *(required)*                                          | Mapillary client token, used to fetch street photos. A token saved in the web app's Settings takes priority.                                    |
| `APP_ENV`                | *(empty)*                                             | Which env file to read (see above).                                                                                                              |
| `SEG_MODEL`              | `facebook/mask2former-swin-large-cityscapes-semantic` | Default Mask2Former checkpoint for the CLI. Any Cityscapes semantic checkpoint works, e.g. `facebook/mask2former-swin-tiny-cityscapes-semantic`. |
| `SEG_DEVICE`             | auto (`cuda` if available, else `cpu`)                | Force the segmentation device: `cuda`, `mps` or `cpu`.                                                                                           |
| `POSM_DATA_DIR`          | `output/jobs`                                         | Web app: where jobs and `settings.json` are kept.                                                                                                |
| `POSM_ENV_DIR`           | the working directory                                 | Where the env file is read from, and created by Settings. Docker Compose sets it to the mounted project folder.                                 |
| `POSM_HOST`              | `127.0.0.1`                                           | Docker Compose only: the address the web app's port is published on. `0.0.0.0` opens it to your network.                                       |

In the Docker images, model weights are downloaded to `/cache` (`HF_HOME` and `TORCH_HOME`), which Docker Compose keeps in the `model-cache` volume.

## Where things are stored

| What                                  | Where                                                                       |
| ------------------------------------- | --------------------------------------------------------------------------- |
| Jobs (model, log, photos, `job.json`) | `output/jobs/<job id>/`                                                     |
| Settings saved in the web app         | `output/jobs/settings.json` (readable by your user only)                    |
| Env file created by Settings          | the project folder (`.env`, or `.env.<APP_ENV>`), readable by your user only |
| Model weights (Docker)                | the `model-cache` Docker volume                                             |
| Model weights (from source)           | the Hugging Face and PyTorch caches in your home folder                     |
