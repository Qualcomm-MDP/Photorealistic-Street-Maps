"""The street-map pipeline, shared by the CLI (main.py) and the web API.

Stage labels and details are what the web UI shows on its progress timeline,
so adding, removing or renaming a stage here is reflected there automatically.
"""

import numpy as np
import trimesh

from common import PipelineChain, export_to_glb
from common.MeshUtils import _get_utm_transformer
from common.reporter import get_reporter
from data_ingest.ingest import ingest_data
from mesh_builder.extrude import build_mesh
from naip import naip_stage
from terrain import terrain_stage
from texturing.tex_projection import tex_projection


def export_mesh(value, state):
    bbox = state.require_metadata("bbox")
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    transformer = _get_utm_transformer(center_lon, center_lat)
    ox, oy = transformer.transform(center_lon, center_lat)

    value.vertices -= np.array([ox, oy, 0.0])
    value.vertices[:, 2] *= -1
    parts = [value]

    terrain = state.get_metadata("terrain", None)
    if terrain is not None:
        ground = terrain.to_mesh()
        naip = state.get_metadata("naip", None)
        if naip is not None:
            ground.visual = naip.drape(ground.vertices)
        ground.vertices -= np.array([ox, oy, 0.0])
        parts.append(ground)

    bounds = np.vstack([p.bounds for p in parts])
    center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.radians(-90.0),
        direction=[1.0, 0.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )
    for p in parts:
        p.apply_translation(-center)
        p.apply_transform(rotation)

    # Buildings and ground stay separate objects: merging them into one mesh
    # would drop the buildings' photo texture.
    if terrain is not None:
        value = trimesh.Scene()
        value.add_geometry(parts[0], node_name="buildings", geom_name="buildings")
        value.add_geometry(parts[1], node_name="terrain", geom_name="terrain")

    # The CLI may pass a callable that asks the user (e.g. a save dialog).
    path = state.get_metadata("output_path", None)
    if path is None:
        ask = state.get_metadata("ask_output_path", None)
        path = ask() if ask else None
    if path is not None:
        export_to_glb(value, path)
        get_reporter(state).info(f"Saved mesh to {path}")

    state.require_metadata("progress_monitor").next()
    return value


def build_pipeline() -> PipelineChain:
    pipeline = PipelineChain("photorealistic-street-maps")
    pipeline.add_stage(
        "fetch", ingest_data, label="Fetch data", detail="OSM + Mapillary"
    )
    pipeline.add_stage(
        "terrain",
        terrain_stage,
        label="Terrain",
        detail="Ground heights (when enabled)",
    )
    pipeline.add_stage(
        "build_mesh", build_mesh, label="Build mesh", detail="Extrude footprints"
    )
    pipeline.add_stage(
        "naip",
        naip_stage,
        label="Aerial photo",
        detail="NAIP ground and roofs (when enabled)",
    )
    pipeline.add_stage(
        "texturing", tex_projection, label="Texture", detail="Project street photos"
    )
    pipeline.add_stage("export", export_mesh, label="Export", detail="Write .glb")
    return pipeline
