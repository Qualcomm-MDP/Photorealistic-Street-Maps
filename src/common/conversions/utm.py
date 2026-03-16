from pyproj import Transformer
import numpy as np
import trimesh

from trimesh import Geometry

from typing import Tuple


def camera_center_utm(image_meta, utm_epsg, camera_height_m):
    lon, lat = image_meta["computed_geometry"]["coordinates"]
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    x, y = t.transform(lon, lat)
    alt = image_meta.get("computed_altitude", camera_height_m)
    return np.array([x, y, alt])


def convert_mesh_to_utm(
    mesh_path: str, lon_and_lat: Tuple[float, float], utm_epsg: int
):
    origin_lon, origin_lat = lon_and_lat[0], lon_and_lat[1]

    loaded = trimesh.load(mesh_path, force="mesh")
    
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("Expected trimesh.Trimesh when loading mesh_path")
    
    mesh = loaded
    
    t_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    t_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    origin_merc_x, origin_merc_y = t_merc.transform(origin_lon, origin_lat)

    vertices = mesh.vertices.copy()
    new_vertices = []

    for v in vertices:
        abs_merc_x = origin_merc_x + v[0]
        abs_merc_y = origin_merc_y + v[1]

        t_merc_to_latlon = Transformer.from_crs(
            "EPSG:3857", "EPSG:4326", always_xy=True
        )
        lon, lat = t_merc_to_latlon.transform(abs_merc_x, abs_merc_y)

        utm_x, utm_y = t_utm.transform(lon, lat)
        new_vertices.append([utm_x, utm_y, v[2]])

    mesh.vertices = np.array(new_vertices)
    return mesh
