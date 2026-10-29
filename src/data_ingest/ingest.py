from common import PipelineState
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES, MapillaryClient
from common.reporter import get_reporter


def ingest_data(bbox: BoundingBox, state: PipelineState):
    reporter = get_reporter(state)
    osm_client = OSMClient(reporter=reporter)
    mapillary_client = MapillaryClient(reporter=reporter)
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
        limit=1000,
    )
    state.require_metadata("progress_monitor").next()

    cumulative_data = {"osm": osm_data, "mapillary": mapillary_data}

    state.set_metadata("provider_data", cumulative_data)
    return cumulative_data
