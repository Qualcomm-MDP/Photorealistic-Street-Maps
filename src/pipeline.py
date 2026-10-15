"""The street-map pipeline, shared by the CLI (main.py) and the web API.

Stage labels and details are what the web UI shows on its progress timeline,
so adding, removing or renaming a stage here is reflected there automatically.
"""

import numpy as np
import trimesh

from common import PipelineChain, export_to_glb
from common.MeshUtils import _get_utm_transformer
from data_ingest.ingest import ingest_data
from mesh_builder.extrude import build_mesh
from texturing.tex_projection import tex_projection


def export_mesh(value, state):
    bbox = state.require_metadata("bbox")
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    transformer = _get_utm_transformer(center_lon, center_lat)
    ox, oy = transformer.transform(center_lon, center_lat)

    value.vertices -= np.array([ox, oy, 0.0])
    value.vertices[:, 2] *= -1
    center = value.bounding_box.centroid
    value.apply_translation(-center)
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.radians(-90.0),
        direction=[1.0, 0.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )
    value.apply_transform(rotation)

    # The CLI may pass a callable that asks the user (e.g. a save dialog).
    path = state.get_metadata("output_path", None)
    if path is None:
        ask = state.get_metadata("ask_output_path", None)
        path = ask() if ask else None
    if path is not None:
        export_to_glb(value, path)
        print(f"Saved mesh to {path}")

    state.require_metadata("progress_monitor").next()
    return value


def build_pipeline() -> PipelineChain:
    pipeline = PipelineChain("photorealistic-street-maps")
    pipeline.add_stage(
        "fetch", ingest_data, label="Fetch data", detail="OSM + Mapillary"
    )
    pipeline.add_stage(
        "build_mesh", build_mesh, label="Build mesh", detail="Extrude footprints"
    )
    pipeline.add_stage(
        "texturing", tex_projection, label="Texture", detail="Project street photos"
    )
    pipeline.add_stage("export", export_mesh, label="Export", detail="Write .glb")
    return pipeline
