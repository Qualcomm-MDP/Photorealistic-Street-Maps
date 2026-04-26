# Stage 4: Evaluation

The evaluation stage measures pipeline performance and output quality. It includes automated profiling of runtime and resource usage, as well as planned qualitative assessment via user surveys.

## Pipeline Profiler

The built-in `PipelineProfiler` instruments each stage with timing and resource measurements. Enable it with the `--profile` flag:

```bash
poetry run python src/main.py --profile results.json
```

### Metrics Collected Per Stage

| Metric                   | Description                                 |
| ------------------------ | ------------------------------------------- |
| `wall_time_s`            | Real elapsed time (includes I/O waits)      |
| `cpu_time_s`             | CPU-only processing time                    |
| `peak_memory_mb`         | Maximum memory traced during the stage      |
| `memory_delta_mb`        | Net memory change (end - start)             |
| `gpu_delta_allocated_mb` | Change in GPU VRAM allocation (NVIDIA only) |
| `status`                 | `ok` or `error`                             |

### Sample Output

The profiler writes a JSON report with system info and per-stage metrics:

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
    {
      "name": "fetch",
      "wall_time_s": 7.607,
      "cpu_time_s": 0.073,
      "peak_memory_mb": 2.5,
      "status": "ok"
    },
    {
      "name": "build_mesh",
      "wall_time_s": 0.196,
      "cpu_time_s": 0.196,
      "peak_memory_mb": 0.6,
      "status": "ok"
    },
    {
      "name": "texturing",
      "wall_time_s": 377.8,
      "cpu_time_s": 286.2,
      "peak_memory_mb": 12440.0,
      "status": "ok"
    },
    {
      "name": "export",
      "wall_time_s": 17.483,
      "cpu_time_s": 3.316,
      "peak_memory_mb": 283.2,
      "status": "ok"
    }
  ]
}
```

It also prints a summary table to the console:

```
Stage                Wall(s)   CPU(s)   Peak MB    GPU Alloc MB   Status
------------------------------------------------------------------------
fetch                   7.61     0.07       2.5           0.0       ok
build_mesh              0.20     0.20       0.6           0.0       ok
texturing             377.80   286.20   12440.0        1060.0       ok
export                 17.48     3.32     283.2           0.0       ok
------------------------------------------------------------------------
TOTAL                 405.30
```

## Cross-Run Comparison

The profiler output is JSON-based, making it straightforward to compare multiple runs across different configurations. The evaluation dashboard supports side-by-side comparison of:

- Runs with vs. without segmentation (`--no-seg`)
- Runs on different hardware (laptop CPU vs. desktop GPU)
- Runs on different region sizes

For example, three benchmark profiles from DR1:

| Profile         | Hardware    | Segmentation | Total Time |
| --------------- | ----------- | ------------ | ---------- |
| `noseg_desktop` | Desktop GPU | Off          | 4m 52s     |
| `seg_desktop`   | Desktop GPU | On           | 6m 43s     |
| `noseg_laptop`  | Laptop CPU  | Off          | 15m 53s    |

Texturing dominates runtime in all configurations, with segmentation adding roughly 2 minutes on desktop GPU.

## Success Criteria

The project defines the following acceptance criteria:

| Criterion          | Target                                    | Measurement                   |
| ------------------ | ----------------------------------------- | ----------------------------- |
| Mesh Generation    | Uncorrupted mesh file                     | Automated validation          |
| Texture Generation | Textures produced from street-view images | Automated validation          |
| Mesh Quality       | Average rating >= 8/10                    | User survey (20 U-M students) |
| Texture Quality    | Average rating >= 8/10                    | User survey (20 U-M students) |
| Total Runtime      | < 25 min per 247 acres                    | Profiler measurement          |
| Usability          | Average rating >= 7/10                    | User survey                   |

!!! info "User Survey"

    Quality metrics are assessed by surveying 20 random University of Michigan students who rate POSM-generated buildings against photographs of their real-world counterparts. This evaluation is in progress.

## Source Code

- `src/common/profiler.py` -- `PipelineProfiler` and `StageMetrics` classes
