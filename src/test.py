from common import ProgressMonitor
from common.constants import BoundingBox
from common.providers import MapillaryClient, OSMClient, OSM_MAP_FEATURES
import time

from mesh_builder.extrude import extrude_buildings

from dotenv import load_dotenv

pm = ProgressMonitor()
pm.add_task("NAME")
pm.add_task()
pm.add_task("TEST")
pm.add_task()

# while True:
#     isFinisehd = pm.next()

#     if isFinisehd:
#         break
#     time.sleep(1)

load_dotenv()

MapillaryClient()

cl = OSMClient()

bbox = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)
data = (cl.fetch(bbox, [OSM_MAP_FEATURES.BUILDING]))

extrude_buildings(data, bbox)