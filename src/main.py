from common import ProgressMonitor, PipelineChain, export_to_glb
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES

from mesh_builder.extrude import extrude_buildings
from data_ingest.ingest import ingest_data 
from mesh_builder.extrude import build_mesh

bbox = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)

def progress():
    # Register a blue print of each task
    progress = ProgressMonitor()
    progress.add_task("Retrive data from OSM")
    progress.add_task("Transform OSM to Mesh data")
    progress.add_task("Export mesh file")
    progress.add_task("FINISHED!")

    # Pipeline
    cl = OSMClient()

    data = (cl.fetch(bbox, [OSM_MAP_FEATURES.BUILDING]))
    progress.next()

    extrude_buildings(data, bbox)
    progress.next()
    progress.next()

    progress.next()

def export_mesh(value, state):
    export_to_glb(value, "combined.glb")

run_pipeline = PipelineChain()
run_pipeline.add_stage("fetech", ingest_data)
run_pipeline.add_stage("build_mesh", build_mesh)
run_pipeline.add_stage("export", export_mesh)

def main():
    run_pipeline.run(bbox, metadata={"bbox": bbox})

if __name__ == "__main__":
    main()