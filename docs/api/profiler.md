# Profiler API

The profiler module provides per-stage instrumentation for the pipeline, tracking wall time, CPU time, memory usage, and GPU VRAM allocation.

## PipelineProfiler

The top-level profiler that wraps an entire pipeline run. Created in `main.py` when the `--profile` flag is passed.

::: common.profiler.PipelineProfiler
options:
show_source: true
members:
\- start
\- begin_stage
\- end_stage
\- finish
\- total_wall_time_s
\- summary
\- save

## StageMetrics

Holds timing and resource measurements for a single pipeline stage. Created by `PipelineProfiler.begin_stage()` and finalized by `end_stage()`.

::: common.profiler.StageMetrics
options:
show_source: true
members:
\- name
\- wall_time_s
\- cpu_time_s
\- peak_memory_mb
\- start_memory_mb
\- end_memory_mb
\- memory_delta_mb
\- gpu_start
\- gpu_end
\- gpu_delta_allocated_mb
\- status
\- error
\- set_throughput

## Usage

### Basic profiling

```python
from common.profiler import PipelineProfiler

profiler = PipelineProfiler(pipeline_name="my-pipeline")
profiler.start()

metrics = profiler.begin_stage("my_stage")
# ... do work ...
profiler.end_stage(metrics, status="ok")

profiler.finish()
profiler.save("performance.json")
```

### With PipelineChain

```python
profiler = PipelineProfiler(pipeline_name="photorealistic-street-maps")

run_pipeline.run(
    bbox,
    metadata={"bbox": bbox, "progress_monitor": progress},
    profiler=profiler,
)

profiler.save("performance.json")
```

The `PipelineChain.run()` method automatically calls `begin_stage` / `end_stage` around each registered stage when a profiler is provided.

## Output Schema

The `summary()` method (and the JSON written by `save()`) follows this structure:

```json
{
  "pipeline": "photorealistic-street-maps",
  "started_at": "2026-03-25T18:00:00+00:00",
  "finished_at": "2026-03-25T18:06:47+00:00",
  "total_wall_time_s": 407.2,
  "system": {
    "python": "3.12.8",
    "os": "Linux-6.5.0-x86_64",
    "cpu_name": "x86_64",
    "cpu_count": 8,
    "gpu": {
      "available": true,
      "name": "NVIDIA GeForce RTX 3070",
      "total_vram_mb": 8192.0,
      "count": 1
    }
  },
  "stages": [
    {
      "name": "fetch",
      "wall_time_s": 7.607,
      "cpu_time_s": 0.0725,
      "peak_memory_mb": 2.5,
      "start_memory_mb": 0.8,
      "end_memory_mb": 1.8,
      "memory_delta_mb": 1.0,
      "gpu_start": {
        "allocated_mb": 0.0
      },
      "gpu_end": {
        "allocated_mb": 0.0
      },
      "gpu_delta_allocated_mb": 0.0,
      "throughput": {},
      "status": "ok",
      "error": null
    }
  ]
}
```

## GPU Detection

The profiler uses `torch.cuda` to detect GPU availability. When no GPU is present, GPU-related fields report `"N/A"` instead of numeric values. The profiler loads `torch` at import time, so it is only suitable for environments where PyTorch is installed.
