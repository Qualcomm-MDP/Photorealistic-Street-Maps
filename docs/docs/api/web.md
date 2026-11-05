---
title: Web API
sidebar_position: 6
---

# Web API

The web app's backend (`src/api/app.py`, FastAPI) serves the frontend from `web/` and the API below. Interactive docs for the running server are at `/api/docs`.

Start it with `docker compose up api`, or from source:

```bash
poetry run uvicorn api.app:create_app --factory --app-dir src
```

## Jobs

| Method | Path                               | Description                                                                                      |
| ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| GET    | `/api/jobs`                        | All jobs, newest first: `{"jobs": [Job, ...]}`                                                   |
| POST   | `/api/jobs`                        | Start a job (see below). Returns the new `Job` with status `201`.                                |
| GET    | `/api/jobs/{id}`                   | One job.                                                                                         |
| GET    | `/api/jobs/{id}/model.glb`         | The model, once the job is done. `?download=true` sends it as a file named after the job.       |
| GET    | `/api/jobs/{id}/log`               | The pipeline log as plain text, one timestamped line per entry. `?download=true` sends a file.  |
| GET    | `/api/jobs/{id}/images`            | The saved photo previews: `{"images": [{"id", "heading", "original", "cleaned"}]}` (URLs).      |
| GET    | `/api/jobs/{id}/images/{name}`     | One preview image (JPEG).                                                                        |

### Starting a job

`POST /api/jobs` with:

```json
{
  "bbox": { "west": -83.71978, "south": 42.29025, "east": -83.71205, "north": 42.29422 },
  "name": "North Campus",
  "remove_obstructions": true,
  "seg_model": "large",
  "device": "auto",
  "terrain": false,
  "naip": false
}
```

| Field                 | Default         | Values                                                                   |
| --------------------- | --------------- | ------------------------------------------------------------------------ |
| `bbox`                | *(required)*    | degrees; west < east, south < north, each side at most 2500 m            |
| `name`                | `"Custom area"` | up to 120 characters                                                     |
| `remove_obstructions` | `true`          | run Mask2Former + LaMa                                                   |
| `seg_model`           | `"large"`       | `"large"` (Swin-Large) or `"tiny"` (Swin-Tiny)                           |
| `device`              | `"auto"`        | `"auto"`, `"cpu"` or `"cuda"` (`422` if the server has no CUDA GPU)      |
| `terrain`             | `false`         | the `--terrain` stage                                                    |
| `naip`                | `false`         | the `--naip` stage; turns `terrain` on                                   |

Invalid requests get `422` with a message in `detail`.

### The Job object

```json
{
  "id": "7c536a830e38",
  "name": "North Campus",
  "bbox": { "west": -83.71978, "south": 42.29025, "east": -83.71205, "north": 42.29422 },
  "options": { "remove_obstructions": true, "seg_model": "large", "device": "auto", "terrain": false, "naip": false },
  "stages": { "fetch": "done", "terrain": "skipped", "build_mesh": "done", "naip": "skipped", "texturing": "running", "export": "pending" },
  "status": "running",
  "error": null,
  "created_at": "2026-11-03T17:18:35+00:00",
  "started_at": "2026-11-03T17:18:35+00:00",
  "finished_at": null,
  "result": {}
}
```

- `status`: `queued`, `running`, `done` or `failed`.
- `stages`: each stage is `pending`, `running`, `done`, `error` or `skipped` (an optional stage that's off for this job).
- `result`, once done: `buildings`, `photos`, `faces`, `model_bytes`, and `thumbnail` (a preview image name).

Jobs run one at a time. A job that was running when the server stopped is marked `failed` when it starts again.

## Live updates

`/api/ws` is a WebSocket. The client doesn't send anything; the server sends JSON messages:

| `type`     | Fields                    | When                                                                                         |
| ---------- | ------------------------- | -------------------------------------------------------------------------------------------- |
| `snapshot` | `jobs`                    | Right after connecting: every job.                                                           |
| `job`      | `job`                     | A job was created, or its status or a stage changed.                                         |
| `log`      | `job_id`, `n`, `line`     | A new pipeline log line. `n` counts from 1, so lines already loaded from `/log` can be skipped. |
| `progress` | `job_id`, `text`          | A progress bar moved (e.g. photo downloads). An empty `text` means it finished.             |

A client that falls too far behind (1000 queued messages) is disconnected; reconnecting gets a fresh `snapshot`.

## Settings

| Method | Path                              | Description                                                                                                   |
| ------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| GET    | `/api/settings`                   | Token status and job defaults (never the token itself).                                                       |
| PUT    | `/api/settings`                   | Change them: `mapillary_token`, `clear_mapillary_token`, `defaults`. Returns the settings plus `token_saved_to`. |
| GET    | `/api/settings/mapillary/token`   | The current token, for the Settings dialog: `{"token": ...}` (not cached).                                   |
| POST   | `/api/settings/mapillary/test`    | Check a token with Mapillary: `{"token": ...}`, or `{}` for the current one. Returns `{"ok", "message"}`.    |

`GET /api/settings` returns:

```json
{
  "mapillary": {
    "set": true,
    "source": "env",
    "hint": "…ab12",
    "env_available": true,
    "env_file": ".env.development",
    "env_file_exists": true
  },
  "defaults": { "remove_obstructions": true, "seg_model": "large", "device": "auto" }
}
```

`source` is `"saved"` (saved in the web app), `"env"` (from the env file or environment) or `null`. `token_saved_to` is `"env_file"` when saving created the env file, otherwise `"settings"`.

## Other endpoints

| Method | Path                                  | Description                                                                                         |
| ------ | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| GET    | `/api/pipeline`                       | The pipeline's stages: `{"stages": [{"name", "label", "detail"}]}`.                                 |
| GET    | `/api/system`                         | `{"cuda", "seg_models", "max_side_m"}`.                                                             |
| GET    | `/api/geocode?q=...`                  | Full address search (Nominatim). At least 3 characters.                                             |
| GET    | `/api/suggest?q=...&lat=...&lon=...`  | Suggestions while typing (Photon), ranked towards `lat`/`lon` when given.                           |

Both searches return `{"results": [...]}`, each with a name, address, `lat`, `lon` and a `bbox`.
