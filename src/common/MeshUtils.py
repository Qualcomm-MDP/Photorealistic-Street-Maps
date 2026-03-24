import trimesh
from trimesh.path import entities

import numpy as np
from pyproj import Transformer
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


def get_height(element):
    height = element["tags"].get("height")

    if not height:
        height = element["tags"].get("building:levels")

        if not height:
            height = float(3 * 1)
        else:
            height = 3 * float(height)
    else:
        height = float(height)

    return height
