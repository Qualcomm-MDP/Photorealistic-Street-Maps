from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES


def ingest_data(value: BoundingBox, state):
    client = OSMClient()
    return client.fetch(value, [OSM_MAP_FEATURES.BUILDING])
