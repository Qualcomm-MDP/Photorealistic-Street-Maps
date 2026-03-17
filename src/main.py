from common import ProgressMonitor, PipelineChain, export_to_glb
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES

from mesh_builder.extrude import extrude_buildings
from data_ingest.ingest import ingest_data
from mesh_builder.extrude import build_mesh
from tkinter import Tk, filedialog


def progress():
    bbox = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)

    # Register a blue print of each task
    progress = ProgressMonitor()
    progress.add_task("Retrieve data from OSM")
    progress.add_task("Transform OSM to Mesh data")
    progress.add_task("Export mesh file")
    progress.add_task("FINISHED!")

    # Pipeline
    cl = OSMClient()

    data = cl.fetch(bbox, [OSM_MAP_FEATURES.BUILDING])
    progress.next()

    extrude_buildings(data, bbox)
    progress.next()
    progress.next()

    progress.next()


# MAIN PIPELINE RUN
pipeline_progress = ProgressMonitor()
pipeline_progress.add_task("Fetching data from OSM ...")
pipeline_progress.add_task("Retrieved data from OSM")
pipeline_progress.add_task("Transform OSM to Mesh data")
pipeline_progress.add_task("Export mesh file")


def ask_save_path(default_name: str = "combined.glb") -> str | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.asksaveasfilename(
        title="Choose where to save",
        defaultextension=".glb",
        initialfile=default_name,
        filetypes=[("GLB files", "*.glb"), ("All files", "*.*")],
    )

    root.destroy()
    return path or None


def export_mesh(value, state):
    path = ask_save_path()
    export_to_glb(value, path or "combined.glb")
    state.metadata["progress_monitor"].next()


run_pipeline = PipelineChain()
run_pipeline.add_stage("fetech", ingest_data)
run_pipeline.add_stage("build_mesh", build_mesh)
run_pipeline.add_stage("export", export_mesh)


def main():
    min_lon = float(input("WEST (Minimum Longitude): "))
    min_lat = float(input("SOUTH (Minimum Latitude): "))
    max_lon = float(input("EAST (Minimum Longitude): "))
    max_lat = float(input("NORTH (Minimum Latitude): "))

    bbox = BoundingBox(min_lat, max_lat, min_lon, max_lon)

    run_pipeline.run(
        bbox, metadata={"bbox": bbox, "progress_monitor": pipeline_progress}
    )


if __name__ == "__main__":
    main()
