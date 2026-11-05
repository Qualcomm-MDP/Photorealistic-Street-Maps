import contextlib
import io
import math
import sys

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
# ...and the alignment fits its cameras to the buildings within 120 m of them
# (pose_alignment/align.py: margin + search + 40 m), so with aligned poses the buildings are
# fetched once over the box grown by this much and handed to it.
ALIGNMENT_OSM_REACH_M = ALIGNMENT_REACH_M + 120.0


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


def overlaps(element: dict, bbox: BoundingBox) -> bool:
    """Whether an OSM element's outline (its geometry, or its members') overlaps the box."""
    pts = element.get("geometry") or [
        p for m in element.get("members", []) for p in m.get("geometry", [])
    ]
    if not pts:
        return False
    lons, lats = [p["lon"] for p in pts], [p["lat"] for p in pts]
    return (
        min(lons) <= bbox.max_lon
        and max(lons) >= bbox.min_lon
        and min(lats) <= bbox.max_lat
        and max(lats) >= bbox.min_lat
    )


def inside(img: dict, bbox: BoundingBox) -> bool:
    lon, lat = img["geometry"]["coordinates"]
    return bbox.min_lon <= lon <= bbox.max_lon and bbox.min_lat <= lat <= bbox.max_lat


class _ReporterLines(io.TextIOBase):
    """A text stream whose lines go to a Reporter (the chooser and the street lookup print
    their report; the web app shows it in the job's log)."""

    def __init__(self, reporter):
        self.reporter, self.buf, self.out = reporter, "", sys.stdout

    def write(self, s: str) -> int:
        self.buf += s
        *lines, self.buf = self.buf.split("\n")
        # the console reporter prints: to the real stdout, not back into this stream
        with contextlib.redirect_stdout(self.out):
            for line in lines:
                if line.strip():
                    self.reporter.info(line)
        return len(s)


def choose_captures(bbox: BoundingBox, records: list[dict], reporter):
    """The sequences to use (data_ingest.mapillary_sequences.choose_roi) along the OSM
    streets in the box (pose_alignment.streets_from_bbox), or None when there are none."""
    from data_ingest.mapillary_sequences import choose_roi
    from pose_alignment.streets_from_bbox import streets_roi

    try:
        with contextlib.redirect_stdout(_ReporterLines(reporter)):
            roi = streets_roi((bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat))
            selection = choose_roi(roi, records)
    except (Exception, SystemExit) as e:
        reporter.warning(
            f"Choosing captures failed ({e}); using every image in the box"
        )
        return None
    if not selection["groups"]:
        reporter.warning(
            "No street has a dense enough capture; using every image in the box"
        )
        return None
    return selection


def ingest_data(bbox: BoundingBox, state: PipelineState):
    """OSM buildings and Mapillary image metadata for the box.

    The photos are the chosen captures (choose_captures): along each OSM street in the
    box, the densest run that covers it, then further runs where that one ends, plus rig
    siblings. Only their frames are downloaded and textured, and with aligned poses
    (pose_source "colmap") the alignment reconstructs the same runs
    (provider_data["mapillary_selection"]). Mapillary is asked for the records over the
    box grown by ALIGNMENT_REACH_M, as the streets' captures run past the box. When no
    capture is usable, every image in the box is used instead.

    Every 360 photo becomes two pinhole views, right and left of travel (data_ingest.pano),
    which texturing cuts out of it.
    """
    reporter = get_reporter(state)
    osm_client = OSMClient(reporter=reporter)
    mapillary_client = MapillaryClient(reporter=reporter)
    state.require_metadata("progress_monitor").next()

    aligning = state.get_metadata("pose_source", "raw") == "colmap"
    if aligning:
        # one request for the mesh's buildings and the alignment's wider ones
        wide = osm_client.fetch(
            grown(bbox, ALIGNMENT_OSM_REACH_M), [OSM_MAP_FEATURES.BUILDING]
        )
        osm_data = {
            **wide,
            "elements": [e for e in wide.get("elements", []) if overlaps(e, bbox)],
        }
    else:
        osm_data = osm_client.fetch(bbox, [OSM_MAP_FEATURES.BUILDING])
    from data_ingest.mapillary_sequences import ALIGNMENT_FIELDS, selection_images

    min_quality = state.get_metadata("min_quality", None)
    cumulative_data = {"osm": osm_data}
    if aligning:
        cumulative_data["osm_alignment"] = wide
    fields = list(dict.fromkeys(MAPILLARY_FIELDS + ALIGNMENT_FIELDS))
    records = mapillary_client.fetch_images(grown(bbox, ALIGNMENT_REACH_M), fields)
    if not any(inside(img, bbox) for img in records):
        raise RuntimeError("No images found.")
    selection = choose_captures(bbox, records, reporter)
    if selection is not None:
        used = selection_images(selection)
        reporter.info(
            f"  Mapillary: {len(used)} frames of {len(selection['groups'])} chosen runs "
            f"(of {len(records)} images near the box)"
        )
        cumulative_data["mapillary_selection"] = selection
    else:
        used = [img for img in records if inside(img, bbox)]
    cumulative_data["mapillary_images"] = records
    cumulative_data["mapillary"] = pano_views(
        mapillary_client.to_metadata(used, min_quality), used
    )
    state.require_metadata("progress_monitor").next()

    state.set_metadata("provider_data", cumulative_data)
    return cumulative_data
