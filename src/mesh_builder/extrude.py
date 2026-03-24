import trimesh
from tqdm import tqdm
import numpy as np
from common.MeshUtils import get_corners, get_lines, get_height


def build_mesh(value, state):
    bbox = state.require_metadata("bbox")
    data = extrude_buildings(value["osm"], bbox)
    state.require_metadata("progress_monitor").next()
    return data


def extrude_buildings(input_data, area_bbox, scale=5):
    buildings = {}

    # _, plane_vertices, plane_faces = initialize_plane(area_bbox, scale)
    # mesh = trimesh.Trimesh(
    #     vertices=plane_vertices,
    #     faces=plane_faces,
    # )

    # buildings = []
    # buildings.append(mesh)

    for _, element in tqdm(
        enumerate(input_data["elements"]),
        "Generating Building Mesh(s)",
        len(input_data["elements"]),
    ):
        if "geometry" not in element:
            continue

        id = element["id"]
        corners = get_corners(element, area_bbox, scale)

        lines = get_lines(corners)

        path = trimesh.path.path.Path2D(
            entities=lines,
            vertices=corners,
        )

        height = get_height(element)

        polys = path.polygons_closed
        if not polys[0]:
            continue

        height = -1 * (height / float(scale))
        if height == 0:
            height += 1e-3

        mesh = path.extrude(height=height)

        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(
                [m.to_mesh() if hasattr(m, "to_mesh") else m for m in mesh]
            )
        else:
            if hasattr(mesh, "to_mesh"):
                mesh = mesh.to_mesh()

        buildings[id] = mesh

    combined_mesh = trimesh.util.concatenate(
        [building for _, building in buildings.items()]
    )
    center = combined_mesh.bounding_box.centroid
    combined_mesh.apply_translation(-center)
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.radians(90.0),
        direction=[1.0, 0.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )

    combined_mesh.apply_transform(rotation)
    return combined_mesh
