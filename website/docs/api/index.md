---
title: API Reference
sidebar_label: Overview
slug: /api
---

# API Reference

Reference for PSM's Python code and the web app's HTTP API. For how the pieces fit together, see [Architecture](../architecture.md); for what each pipeline stage does, see [Pipeline](../pipeline/overview.md).

## Core

| Page | What's in it |
| --- | --- |
| [Pipeline & State](pipeline.md) | `PipelineChain` and `PipelineState`, the PSM pipeline itself (`build_pipeline`, export) and `BoundingBox`. |
| [Data Providers](providers.md) | The OpenStreetMap (Overpass) and Mapillary clients, and the fetch stage. |
| [Utilities](utilities.md) | Mesh building, terrain, NAIP aerial photos, texturing, obstruction removal, mesh helpers and export, coordinate conversions, progress display. |
| [Reporter](reporter.md) | How pipeline code reports output: `Reporter`, the web app's `JobReporter`, and keeping secrets out of logs. |
| [Profiler](profiler.md) | `PipelineProfiler` and `StageMetrics`, behind `--profile`. |

These pages are generated from the docstrings in `src/`. To update them after changing a docstring, run `poetry run npm run api --prefix website` from the repo root and commit the result. The published site is always generated fresh, so it's never out of date.

## Web app

| Page | What's in it |
| --- | --- |
| [Web API](web.md) | The REST endpoints (jobs, settings, search), the job object, and the WebSocket messages for live updates. |

A running server also serves interactive docs at `/api/docs`.
