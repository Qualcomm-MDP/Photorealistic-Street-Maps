from common import ProgressMonitor
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES

from mesh_builder.extrude import extrude_buildings

def main():
    # Register a blue print of each task
    progress = ProgressMonitor()
    progress.add_task("Retrive data from OSM")
    progress.add_task("Transform OSM to Mesh data")
    progress.add_task("Export mesh file")
    progress.add_task("FINISHED!")

    # Pipeline
    cl = OSMClient()

    bbox = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)
    data = (cl.fetch(bbox, [OSM_MAP_FEATURES.BUILDING]))
    progress.next()

    extrude_buildings(data, bbox)
    progress.next()
    progress.next()

    progress.next()


if __name__ == "__main__":
    main()