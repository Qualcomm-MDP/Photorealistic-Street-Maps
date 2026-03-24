from common import PipelineState
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES, MapillaryClient


def ingest_data(bbox: BoundingBox, state: PipelineState):
    osm_client = OSMClient()
    mapillary_client = MapillaryClient()
    state.require_metadata("progress_monitor").next()

    osm_data = osm_client.fetch(bbox, [OSM_MAP_FEATURES.BUILDING])
    mapillary_data = mapillary_client.fetch(
        bbox,
        [
            "id",
            "thumb_original_url",
            "sequence_id",
            "geometry",
            "computed_geometry",
            "computed_compass_angle",
            "computed_rotation",
            "camera_parameters",
            "width",
            "height",
        ],
    )
    state.require_metadata("progress_monitor").next()

    cumulative_data = {"osm": osm_data, "mapillary": mapillary_data}

    state.set_metadata("provider_data", cumulative_data)
    return cumulative_data
