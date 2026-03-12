import trimesh
from trimesh.path import entities 

import numpy as np
from .constants import BoundingBox

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
    max_lat = int(bounding_box.max_lat * (10**Scale))
    min_lat = int(bounding_box.min_lat * (10**Scale))
    max_lon = int(bounding_box.max_lon * (10**Scale))
    min_lon = int(bounding_box.min_lon * (10**Scale))

    delta_lat = abs(max_lat - min_lat) 
    delta_long = abs(max_lon - min_lon) 
    plane, corners, faces = generate_plane(delta_lat, delta_long) 

    return plane, corners, faces

def get_corners(element, bounding_box: BoundingBox, Scale: int):
    geometry_points = element["geometry"]
    corners = []

    for point in geometry_points:
        latitude = int(float(point["lat"]) * (10**Scale))
        longitude = int(float(point["lon"]) * (10**Scale))

        local_i = abs(latitude - int(bounding_box.min_lat * (10**Scale)))
        local_j = abs(longitude - int(bounding_box.min_lon * (10**Scale)))

        corners.append([local_i, local_j])
    
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
