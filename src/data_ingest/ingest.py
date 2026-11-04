import math

from common import PipelineState
from common.constants import BoundingBox
from common.providers import OSMClient, OSM_MAP_FEATURES, MapillaryClient
from common.reporter import get_reporter
from data_ingest.pano import pano_views

MAPILLARY_FIELDS = [
    "id",
    "thumb_original_url",
    "sequence_id",
    "geometry",
    "computed_geometry",
    "computed_compass_angle",
    "computed_rotation",
    "camera_parameters",
    "camera_type",
    # 360 photos' direction of travel (data_ingest.pano)
    "sequence",
    "captured_at",
    "compass_angle",
    "width",
    "height",
    "quality_score",
]
# How far past the box the pose alignment looks for captures (m): its street lines run
# 25 m past the box (pose_alignment.streets_from_bbox) and it takes images up to 40 m from
# a line (15 m corridor + 25 m slack; data_ingest.mapillary_sequences.find_runs).
ALIGNMENT_REACH_M = 70.0


def grown(bbox: BoundingBox, metres: float) -> BoundingBox:
    d_lat = metres / 111320.0
    d_lon = metres / (
        111320.0 * math.cos(math.radians((bbox.min_lat + bbox.max_lat) / 2))
    )
    return BoundingBox(
        bbox.min_lat - d_lat,
        bbox.max_lat + d_lat,
        bbox.min_lon - d_lon,
        bbox.max_lon + d_lon,
    )


def inside(img: dict, bbox: BoundingBox) -> bool:
    lon, lat = img["geometry"]["coordinates"]
    return bbox.min_lon <= lon <= bbox.max_lon and bbox.min_lat <= lat <= bbox.max_lat


def ingest_data(bbox: BoundingBox, state: PipelineState):
    """OSM buildings and Mapillary image metadata for the box.

    Every 360 photo becomes two pinhole views, right and left of travel (data_ingest.pano),
    which texturing cuts out of it.

    With aligned poses (pose_source "colmap"), Mapillary is asked for the fields the pose
    alignment needs too, over the box grown by ALIGNMENT_REACH_M; those records are kept as
    provider_data["mapillary_images"], and the alignment picks its captures from them
    instead of querying Mapillary again. Texturing still gets only the box's images.
    """
    reporter = get_reporter(state)
    osm_client = OSMClient(reporter=reporter)
    mapillary_client = MapillaryClient(reporter=reporter)
    state.require_metadata("progress_monitor").next()

    osm_data = osm_client.fetch(bbox, [OSM_MAP_FEATURES.BUILDING])
    min_quality = state.get_metadata("min_quality", None)
    cumulative_data = {"osm": osm_data}
    if state.get_metadata("pose_source", "raw") == "colmap":
        from data_ingest.mapillary_sequences import ALIGNMENT_FIELDS

        fields = list(dict.fromkeys(MAPILLARY_FIELDS + ALIGNMENT_FIELDS))
        records = mapillary_client.fetch_images(grown(bbox, ALIGNMENT_REACH_M), fields)
        in_box = [img for img in records if inside(img, bbox)]
        if not in_box:
            raise RuntimeError("No images found.")
        cumulative_data["mapillary_images"] = records
    else:
        in_box = mapillary_client.fetch_images(bbox, MAPILLARY_FIELDS)
        if not in_box:
            raise RuntimeError("No images found.")
    cumulative_data["mapillary"] = pano_views(
        mapillary_client.to_metadata(in_box, min_quality), in_box
    )
    state.require_metadata("progress_monitor").next()

    state.set_metadata("provider_data", cumulative_data)
    return cumulative_data
