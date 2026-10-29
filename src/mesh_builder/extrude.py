import trimesh
from common.MeshUtils import get_corners, get_lines, get_height
from common.reporter import Reporter, get_reporter


def build_mesh(value, state):
    bbox = state.require_metadata("bbox")
    data = extrude_buildings(value["osm"], bbox, reporter=get_reporter(state))
    state.require_metadata("progress_monitor").next()
    return data


def extrude_buildings(input_data, area_bbox, reporter: Reporter | None = None):
    reporter = reporter or get_reporter()
    buildings = {}

    for _, element in reporter.progress(
        enumerate(input_data["elements"]),
        desc="Generating Building Mesh(s)",
        total=len(input_data["elements"]),
    ):
        if "geometry" not in element:
            continue

        id = element["id"]
        corners = get_corners(element, area_bbox)

        lines = get_lines(corners)

        path = trimesh.path.path.Path2D(
            entities=lines,
            vertices=corners,
        )

        height = get_height(element)

        polys = path.polygons_closed
        if not polys[0]:
            continue

        height = -1 * height
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
    return combined_mesh
