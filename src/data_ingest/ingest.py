from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES


def ingest_data(value: BoundingBox, state):
    client = OSMClient()
    state.metadata["progress_monitor"].next()
    data = client.fetch(value, [OSM_MAP_FEATURES.BUILDING])
    state.metadata["progress_monitor"].next()
    return data