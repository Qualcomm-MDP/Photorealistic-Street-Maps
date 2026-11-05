# Web App

The web app runs the same pipeline as the command line, with a map to pick the area, live progress, a 3D preview and a history of jobs. Start it with `docker compose up --build api` and open <http://localhost:8000> (see [Getting Started](getting-started.mdx)).

## Picking an area

- **Search** for an address or place. Suggestions appear as you type; press **Search** for a full search. Choosing a result frames a sensible area around it.
- **Draw area**: press it, then drag a box on the map. **Esc** cancels.
- **Coordinates**: type west, south, east and north directly.
- **Open larger map** shows the same map in a bigger dialog.

The size of the area is shown under the coordinates. Each side can be at most 2.5 km.

![The Area panel: address search, the OSM map with the selected area drawn on it, Generate model, and the area's coordinates and size](/img/screenshots/pick-area.webp)

## Options

| Option                  | What it does                                                                                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Remove obstructions** | Erase cars, people, poles and similar from the street photos before projecting them (Mask2Former + LaMa). Slower, cleaner facades.                                                     |
| **Segmentation model**  | Mask2Former **Swin-Large** (best quality) or **Swin-Tiny** (much faster on a CPU).                                                                                                       |
| **Device**              | **Auto** uses an NVIDIA GPU when the server has one, otherwise the CPU. **GPU** is only available on the GPU image with a GPU.                                                          |
| **Terrain**             | Put buildings on real ground heights and add the ground to the model.                                                                                                                   |
| **Aerial photos**       | Texture the ground and roofs with NAIP aerial photos. Continental US only; turns on Terrain (and locks it on while it's on). A note appears if the area looks outside NAIP's coverage. |
| **Profile this run**    | Measure each stage's time and memory, shown live in the job's details. See [Profiling a job](#profiling-a-job).                                                                         |

The starting values come from the defaults in [Settings](#settings).

![The Options panel: camera poses, Aligned photos only, Remove obstructions, segmentation model, device, Terrain and Aerial photos](/img/screenshots/options.webp)

## Running a job

Press **Generate model**. While a job runs:

- The **timeline** shows each pipeline stage as pending, running, done, or failed. Stages turned off for this job (Terrain, Aerial photo) show as **Skipped**.
- The **pipeline log** streams the job's output live, including progress bars such as the photo downloads. **Download log** saves it as a text file.
- The 3D preview is greyed out, and **Generate model** is disabled until the job finishes.

Jobs run one at a time. Finished jobs survive a page reload or a server restart; a job that was still running when the server stopped is marked as failed.

![A job running: the 3D preview greyed out with its current stage, the timeline with the stages done so far and the skipped ones, and the live pipeline log](/img/screenshots/job-running.webp)

When the job finishes, the model loads into the **3D preview** (drag to orbit, scroll to zoom). **.glb** downloads it, and **Larger view** opens a bigger viewer. The **Street imagery** section shows the first 12 photos of the job, before and after obstruction removal.

![The finished model in the 3D preview, with the completed timeline and the pipeline log below](/img/screenshots/preview.webp)

![The Street imagery section: the job's street photos, with the Original and Cleaned toggle](/img/screenshots/street-imagery.webp)

## Recent jobs

Every job is listed with its area, status, building count and duration. Click a row to open its details:

![The Recent jobs table: each job's area, bounding box, status, building count and duration](/img/screenshots/recent-jobs.webp)


- the model, with stats and a download button
- the job's settings (area, options, device) and its error, if it failed
- the full pipeline log
- the street photos


![A job's details: the model, its stats and download, the job's settings, the pipeline log and its photos](/img/screenshots/job-details.webp)

Jobs are saved under `output/jobs/<job id>/` (in Docker, the `output` folder in your project) with `job.json`, `pipeline.log`, `model.glb` and the photo previews.

## Profiling a job

Turn on **Profile this run** in the Options panel to measure each pipeline stage's time and memory. The job's details then get a **Profile** section:

- **Headline figures:** total time, the slowest stage and its share of the run, the highest memory peak and where it happened, and CPU use (CPU time divided by wall time; over 100% means several cores were busy).
- **Time per stage** and **Peak memory per stage:** one bar per stage, with its value at the end. Hover or focus a bar for its wall time, CPU time and peak memory. Stages that are off for the job are labelled **skipped**; a stage that failed has a red bar marked **⚠ error**, and its error is in the tooltip.
- **All metrics:** a table with every stage's wall time, CPU time, CPU %, peak memory, memory change and GPU memory.
- **Download JSON** saves the full report (`profile.json`, the same format as the command line's `--profile`, see [Evaluation](pipeline/evaluation.md#pipeline-profiler)).

The profile **updates live** while the job runs: after every stage, and the running stage's time ticks every second. You can open it at any point:

- **View profile**, next to the progress bar, opens the current job's profile.
- In **Recent jobs**, profiled jobs carry a **Profiled** marker; open the job to see its profile, finished or not.

A job that fails keeps its profile up to the failed stage. Peak memory is what Python's `tracemalloc` sees: Python and NumPy allocations, not the whole process (for example not PyTorch's own buffers or GPU memory). Profiling makes a run slower, because `tracemalloc` records every allocation.

## Settings

The gear icon in the header opens Settings.

![The Settings dialog: the Mapillary token's status and masked value with Change and Validate Token, and the defaults for new jobs](/img/screenshots/settings.webp)

### Mapillary access token

- The status shows whether a token is set and where it comes from: **Saved here**, or **From .env** (the env file for your `APP_ENV`, e.g. `.env.development`), with its last four characters.
- The token shows as dots; **Show** reveals it.
- Press **Change**, paste the new token, then **Save token**. PSM asks you to confirm before it replaces the current one. New jobs use it straight away.
- **Validate Token** checks a token with Mapillary.
- **Remove saved token** (or **Use the .env token instead**) removes the one saved here, after confirming.

Where a saved token goes:

- If the env file doesn't exist yet, saving creates it (e.g. `.env` in your project folder) with `MAPILLARY_ACCESS_TOKEN=...`.
- Otherwise the token goes into `output/jobs/settings.json` and takes priority over the env file's.

Both files are readable by your user only. See [Configuration](configuration.md).

### Defaults for new jobs

Obstruction removal, segmentation model and device. They fill the Options panel each time PSM opens.

Without a token, a notice next to **Generate model** links to Settings, and pressing Generate opens Settings instead of starting a job that would fail.

## Who can reach it

The server listens on `127.0.0.1` (this computer only) by default. It has no login, and Settings can show the Mapillary token, so only open it to your network if you trust everyone on it:

```bash
PSM_HOST=0.0.0.0 docker compose up api
```

## Light and dark mode

The sun/moon button switches themes; the choice is remembered in the browser. By default the app follows the system setting.

![The web app in dark mode](/img/screenshots/app-overview-dark.webp)
