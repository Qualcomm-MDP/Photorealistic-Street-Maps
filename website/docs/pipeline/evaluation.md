# Evaluation

POSM is evaluated on speed and resource use, measured by the built-in profiler, and on output quality, assessed with user surveys.

## Pipeline profiler

`PipelineProfiler` (`src/common/profiler.py`) measures each stage. Turn it on from the command line:

```bash
poetry run python src/main.py --bbox W S E N --output map.glb --profile results.json
```

### Metrics per stage

| Metric                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `wall_time_s`            | Real elapsed time (includes waiting on downloads) |
| `cpu_time_s`             | CPU time used                                     |
| `peak_memory_mb`         | Peak Python memory traced during the stage        |
| `memory_delta_mb`        | Memory at the end minus at the start              |
| `gpu_delta_allocated_mb` | Change in allocated GPU memory (NVIDIA only)      |
| `status`                 | `ok` or `error`                                   |

### Output

The profiler writes a JSON report with system information and the per-stage metrics, and prints a summary table. The example below is from **Design Review 1**, when the pipeline had four stages; today the report also lists `terrain` and `naip`.

```json
{
  "pipeline": "photorealistic-street-maps",
  "started_at": "2026-03-25T18:00:00Z",
  "total_wall_time_s": 405.3,
  "system": {
    "python": "3.12.0",
    "os": "Linux-6.1.0-x86_64",
    "cpu_name": "x86_64",
    "cpu_count": 8,
    "gpu": {
      "available": true,
      "name": "NVIDIA RTX 4090",
      "total_vram_mb": 24564.0,
      "count": 1
    }
  },
  "stages": [
    { "name": "fetch", "wall_time_s": 7.607, "cpu_time_s": 0.073, "peak_memory_mb": 2.5, "status": "ok" },
    { "name": "build_mesh", "wall_time_s": 0.196, "cpu_time_s": 0.196, "peak_memory_mb": 0.6, "status": "ok" },
    { "name": "texturing", "wall_time_s": 377.8, "cpu_time_s": 286.2, "peak_memory_mb": 12440.0, "status": "ok" },
    { "name": "export", "wall_time_s": 17.483, "cpu_time_s": 3.316, "peak_memory_mb": 283.2, "status": "ok" }
  ]
}
```

```
Stage                 Wall(s)   CPU(s)   Peak MB  GPU Alloc MB   Status
------------------------------------------------------------------------
fetch                    7.61     0.07       2.5           0.0       ok
build_mesh               0.20     0.20       0.6           0.0       ok
texturing              377.80   286.20   12440.0        1060.0       ok
export                  17.48     3.32     283.2           0.0       ok
------------------------------------------------------------------------
TOTAL                  405.30
```

## Comparing runs

The JSON reports make it easy to compare runs with and without obstruction removal (`--no-seg`), on different hardware, or on different area sizes. Three benchmark runs from **Design Review 1** (older pipeline):

| Profile         | Hardware    | Segmentation | Total Time |
| --------------- | ----------- | ------------ | ---------- |
| `noseg_desktop` | Desktop GPU | Off          | 4m 52s     |
| `seg_desktop`   | Desktop GPU | On           | 6m 43s     |
| `noseg_laptop`  | Laptop CPU  | Off          | 15m 53s    |

Texturing dominated the runtime in all three, and segmentation added about 2 minutes on the desktop GPU.

## Success criteria

| Criterion          | Target                                    | Measurement                   |
| ------------------ | ----------------------------------------- | ----------------------------- |
| Mesh generation    | Uncorrupted mesh file                     | Automated validation          |
| Texture generation | Textures produced from street-view images | Automated validation          |
| Mesh quality       | Average rating ≥ 8/10                     | User survey (20 U-M students) |
| Texture quality    | Average rating ≥ 8/10                     | User survey (20 U-M students) |
| Total runtime      | < 25 min per 247 acres                    | Profiler                      |
| Usability          | Average rating ≥ 7/10                     | User survey                   |

:::info[User survey]

Quality is assessed by asking 20 randomly chosen University of Michigan students to rate POSM buildings against photographs of the real ones. This evaluation is in progress.

:::

## Source code

- `src/common/profiler.py`: `PipelineProfiler`, `StageMetrics`
