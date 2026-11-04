# Web App

The web app runs the same pipeline as the command line, with a map to pick the area, live progress, a 3D preview and a history of jobs. Start it with `docker compose up --build api` and open <http://localhost:8000> (see [Getting Started](getting-started.md)).

## Picking an area

- **Search** for an address or place. Suggestions appear as you type; press **Search** for a full search. Choosing a result frames a sensible area around it.
- **Draw area**: press it, then drag a box on the map. **Esc** cancels.
- **Coordinates**: type west, south, east and north directly.
- **Open larger map** shows the same map in a bigger dialog.

The size of the area is shown under the coordinates. Each side can be at most 2.5 km.

## Options

| Option                  | What it does                                                                                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Remove obstructions** | Erase cars, people, poles and similar from the street photos before projecting them (Mask2Former + LaMa). Slower, cleaner facades.                                                     |
| **Segmentation model**  | Mask2Former **Swin-Large** (best quality) or **Swin-Tiny** (much faster on a CPU).                                                                                                       |
| **Device**              | **Auto** uses an NVIDIA GPU when the server has one, otherwise the CPU. **GPU** is only available on the GPU image with a GPU.                                                          |
| **Terrain**             | Put buildings on real ground heights and add the ground to the model.                                                                                                                   |
| **Aerial photos**       | Texture the ground and roofs with NAIP aerial photos. Continental US only; turns on Terrain (and locks it on while it's on). A note appears if the area looks outside NAIP's coverage. |

The starting values come from the defaults in [Settings](#settings).

## Running a job

Press **Generate model**. While a job runs:

- The **timeline** shows each pipeline stage as pending, running, done, or failed. Stages turned off for this job (Terrain, Aerial photo) show as **Skipped**.
- The **pipeline log** streams the job's output live, including progress bars such as the photo downloads. **Download log** saves it as a text file.
- The 3D preview is greyed out, and **Generate model** is disabled until the job finishes.

Jobs run one at a time. Finished jobs survive a page reload or a server restart; a job that was still running when the server stopped is marked as failed.

When the job finishes, the model loads into the **3D preview** (drag to orbit, scroll to zoom). **.glb** downloads it, and **Larger view** opens a bigger viewer. The **Street imagery** section shows the first 12 photos of the job, before and after obstruction removal.

## Recent jobs

Every job is listed with its area, status, building count and duration. Click a row to open its details:

- the model, with stats and a download button
- the job's settings (area, options, device) and its error, if it failed
- the full pipeline log
- the street photos

Jobs are saved under `output/jobs/<job id>/` (in Docker, the `output` folder in your project) with `job.json`, `pipeline.log`, `model.glb` and the photo previews.

## Settings

The gear icon in the header opens Settings.

### Mapillary access token

- The status shows whether a token is set and where it comes from: **Saved here**, or **From .env** (the env file for your `APP_ENV`, e.g. `.env.development`), with its last four characters.
- The token shows as dots; **Show** reveals it.
- Press **Change**, paste the new token, then **Save token**. POSM asks you to confirm before it replaces the current one. New jobs use it straight away.
- **Validate Token** checks a token with Mapillary.
- **Remove saved token** (or **Use the .env token instead**) removes the one saved here, after confirming.

Where a saved token goes:

- If the env file doesn't exist yet, saving creates it (e.g. `.env` in your project folder) with `MAPILLARY_ACCESS_TOKEN=...`.
- Otherwise the token goes into `output/jobs/settings.json` and takes priority over the env file's.

Both files are readable by your user only. See [Configuration](configuration.md).

### Defaults for new jobs

Obstruction removal, segmentation model and device. They fill the Options panel each time POSM opens.

Without a token, a notice next to **Generate model** links to Settings, and pressing Generate opens Settings instead of starting a job that would fail.

## Who can reach it

The server listens on `127.0.0.1` (this computer only) by default. It has no login, and Settings can show the Mapillary token, so only open it to your network if you trust everyone on it:

```bash
POSM_HOST=0.0.0.0 docker compose up api
```

## Light and dark mode

The sun/moon button switches themes; the choice is remembered in the browser. By default the app follows the system setting.
