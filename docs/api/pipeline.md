# Pipeline & State API

The pipeline framework provides the core orchestration layer for POSM. It defines how stages are registered, chained, and executed with shared state.

## PipelineChain

The `PipelineChain` class manages an ordered sequence of stages and executes them in order, passing each stage's output as the next stage's input.

::: common.pipeline_chain.PipelineChain
options:
show_source: true
members:
\- add_stage
\- add_fork
\- run
\- resume
\- stage_names

### Usage Example

```python
from common.pipeline_chain import PipelineChain

pipeline = PipelineChain(name="my-pipeline")
pipeline.add_stage("fetch", my_fetch_fn)
pipeline.add_stage("process", my_process_fn)
pipeline.add_stage("export", my_export_fn)

state = pipeline.run(
    initial_input=bounding_box,
    metadata={"key": "value"},
)
```

### Resuming from a Stage

If a pipeline fails partway through, you can resume from any stage using the state object returned by the failed run:

```python
state = pipeline.resume(state, from_stage="process")
```

## PipelineState

`PipelineState` carries the running context through the pipeline. Stages read the current value, access metadata, and store outputs.

::: common.pipeline_chain.PipelineState
options:
show_source: true
members:
\- get_metadata
\- require_metadata
\- set_metadata
\- get_output
\- set_output

### Key Metadata Keys

These metadata keys are set by `main.py` and used across stages:

| Key                   | Type              | Set By        | Used By                                       |
| --------------------- | ----------------- | ------------- | --------------------------------------------- |
| `bbox`                | `BoundingBox`     | `main.py`     | `build_mesh`, `tex_projection`, `export_mesh` |
| `progress_monitor`    | `ProgressMonitor` | `main.py`     | All stages                                    |
| `remove_obstructions` | `bool`            | `main.py`     | `tex_projection`                              |
| `provider_data`       | `dict`            | `ingest_data` | `tex_projection`                              |

## PipelineProfiler

The profiler wraps each stage with timing and resource instrumentation.

::: common.profiler.PipelineProfiler
options:
show_source: true
members:
\- start
\- begin_stage
\- end_stage
\- finish
\- save
\- summary

## StageMetrics

Per-stage measurement data collected by the profiler.

::: common.profiler.StageMetrics
options:
show_source: true
members:
\- set_throughput

## BoundingBox

The core input type representing a geographic region.

::: common.constants.BoundingBox
options:
show_source: true
