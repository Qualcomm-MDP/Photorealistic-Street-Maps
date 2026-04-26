# Pipeline Overview

The POSM pipeline transforms a geographic bounding box into a photorealistic textured 3D mesh in four sequential stages. Each stage is a standalone function wired together by the `PipelineChain` framework.

## End-to-End Flow

```mermaid
flowchart LR
    IN["GPS Bounding Box"] --> S1["1. Data Processing"]
    S1 --> S2["2. Mesh Generation"]
    S2 --> S3["3. Texturing"]
    S3 --> S4["4. Evaluation"]
    S4 --> OUT[".glb File +<br/>Metrics JSON"]

    style S1 fill:#3355ff,color:#fff
    style S2 fill:#3355ff,color:#fff
    style S3 fill:#3355ff,color:#fff
    style S4 fill:#3355ff,color:#fff
```

## Stage Summary

| Stage                  | Input                                  | Output                                                                                 | Key Technologies                                                       |
| ---------------------- | -------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **1. Data Processing** | Bounding box coordinates               | Building geometry (OSM JSON) + street-view images (Mapillary) + satellite tiles (NAIP) | Overpass API, Mapillary API, NAIP STAC, Mask2Former, phase correlation |
| **2. Mesh Generation** | Building geometry data                 | Combined 3D trimesh (untextured)                                                       | Trimesh path extrusion, UTM coordinate transforms                      |
| **3. Texturing**       | 3D mesh + street-view/satellite images | Textured 3D mesh with atlas-packed UV maps                                             | OpenCV homography, camera projection, LaMa inpainting                  |
| **4. Evaluation**      | Textured mesh + profiler data          | Performance metrics JSON, user survey results                                          | PipelineProfiler (wall time, CPU, memory, GPU VRAM)                    |

## Pipeline Wiring

The stages are registered in `src/main.py`:

```python
from common import PipelineChain

run_pipeline = PipelineChain()
run_pipeline.add_stage("fetch", ingest_data)
run_pipeline.add_stage("build_mesh", build_mesh)
run_pipeline.add_stage("texturing", tex_projection)
run_pipeline.add_stage("export", export_mesh)
```

Each stage function has the signature `(value, state: PipelineState) -> value`, where `value` is the output of the previous stage and `state` provides access to shared metadata like the bounding box and progress monitor.

## Data Flow Between Stages

```mermaid
flowchart TD
    subgraph fetch["fetch"]
        direction TB
        F1["OSM Client → building footprints"]
        F2["Mapillary Client → street-view metadata + URLs"]
        F1 & F2 --> FO["dict: osm + mapillary"]
    end

    subgraph build_mesh["build_mesh"]
        BM["Extrude each OSM footprint<br/>into a 3D solid"] --> BMO["trimesh.Trimesh<br/>combined, untextured"]
    end

    subgraph texturing["texturing"]
        T1["Download images from Mapillary URLs"]
        T2["Project each image onto visible mesh faces"]
        T3["Remove obstructions via Mask2Former + LaMa"]
        T4["Pack face patches into texture atlas"]
        T1 --> T2 --> T3 --> T4 --> TO["trimesh.Trimesh<br/>textured, with UV + atlas"]
    end

    subgraph export["export"]
        E1["Center mesh at origin"]
        E2["Flip Z axis, rotate -90 deg around X"]
        E3["Save as .glb via file dialog"]
        E1 --> E2 --> E3
    end

    fetch --> build_mesh --> texturing --> export
```

## Running Modes

The pipeline supports two optional flags that affect behavior:

- **`--profile <filename>`** — wraps each stage with `PipelineProfiler`, writing per-stage timing and resource usage to a JSON file.
- **`--no-seg`** — skips the Mask2Former + LaMa obstruction-removal pass in the texturing stage, producing faster but noisier textures.

## Success Criteria

| Metric             | Target                                                        |
| ------------------ | ------------------------------------------------------------- |
| Mesh generation    | Uncorrupted `.glb` file                                       |
| Texture generation | Textures derived from street-view images                      |
| Mesh quality       | Average user survey rating ≥ 8/10 on building shape realism   |
| Texture quality    | Average user survey rating ≥ 8/10 on building texture realism |
| Total runtime      | Less than 25 minutes per 247 acres                            |
| Usability          | Average user rating ≥ 7/10 on pipeline ease of use            |
