import numpy as np
import trimesh
from pyproj import Transformer
from trimesh.path import entities

from .constants import BoundingBox


def _get_utm_transformer(lon: float, lat: float) -> Transformer:
    zone_number = int((lon + 180) / 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone_number
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def _scale_distance(value: float, scale: int) -> float:
    if scale <= 0:
        raise ValueError("Scale must be greater than 0")
    return value / float(scale)


# Follow https://github.com/Qualcomm-MDP/mesh_creation/blob/main/extrude_out.py
def generate_plane(height, width):
    corners = [
        [0, 0, 0],
        [0, width, 0],
        [height, width, 0],
        [height, 0, 0],
    ]

    faces = np.array([[0, 1, 2, 3]])
    plane = trimesh.Trimesh(vertices=corners, faces=faces)

    return plane, corners, faces


def initialize_plane(bounding_box: BoundingBox, Scale: int):
    center_lon = (bounding_box.min_lon + bounding_box.max_lon) / 2
    center_lat = (bounding_box.min_lat + bounding_box.max_lat) / 2
    transformer = _get_utm_transformer(center_lon, center_lat)

    min_x, min_y = transformer.transform(
        bounding_box.min_lon,
        bounding_box.min_lat,
    )
    max_x, max_y = transformer.transform(
        bounding_box.max_lon,
        bounding_box.max_lat,
    )

    delta_x = _scale_distance(max_x - min_x, Scale)
    delta_y = _scale_distance(max_y - min_y, Scale)
    plane, corners, faces = generate_plane(delta_x, delta_y)

    return plane, corners, faces


def get_corners(element, bounding_box: BoundingBox):
    geometry_points = element["geometry"]
    corners = []

    center_lon = (bounding_box.min_lon + bounding_box.max_lon) / 2
    center_lat = (bounding_box.min_lat + bounding_box.max_lat) / 2
    transformer = _get_utm_transformer(center_lon, center_lat)

    for point in geometry_points:
        projected_x, projected_y = transformer.transform(
            float(point["lon"]),
            float(point["lat"]),
        )
        corners.append([projected_x, projected_y])

    return corners


def get_lines(corners, loop=True):
    lines = []
    start = 0
    end = 1
    lines.append(entities.Line([start, end]))

    for i in range(len(corners) - 2):
        start += 1
        end += 1
        lines.append(entities.Line([start, end]))

    if loop:
        lines.append(entities.Line([end, 0]))
    return lines


# Building heights from OSM tags (shared with the pose alignment's fit to the footprints).
LEVEL_HEIGHT_M = 3.5  # per building:levels / building:min_level
DEFAULT_LEVELS = 3  # an untagged building: 3 storeys, 10.5 m
MIN_WALL_M = 2.5  # a part is at least this tall above its base


def parse_metres(value):
    """A tag value in metres ("12", "12 m", "12.5;14"), or None."""
    try:
        return float(str(value).lower().replace("m", "").split(";")[0].strip())
    except ValueError:
        return None


def building_heights(tags, level_height=LEVEL_HEIGHT_M, default_levels=DEFAULT_LEVELS):
    """(base, top, source) in metres above ground from OSM tags: height / building:height,
    else building:levels x level_height, else default_levels storeys; the base from
    min_height or building:min_level (a part that starts above the ground)."""
    base = parse_metres(tags.get("min_height"))
    if base is None and tags.get("building:min_level"):
        base = (parse_metres(tags["building:min_level"]) or 0) * level_height
    top = parse_metres(tags.get("height") or tags.get("building:height"))
    source = "height"
    if top is None and tags.get("building:levels"):
        levels = parse_metres(tags["building:levels"])
        if levels:
            top, source = levels * level_height, "levels"
    if top is None:
        top, source = default_levels * level_height, "default"
    return base or 0.0, top, source


def get_base_and_height(element):
    """(base, top) in metres above ground of an OSM building element (building_heights)."""
    base, top, _ = building_heights(element.get("tags", {}))
    return base, max(top, base + MIN_WALL_M)


def get_height(element):
    return get_base_and_height(element)[1]
