import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon

from common.MeshUtils import get_base_and_height, get_corners, get_lines
from common.reporter import Reporter, get_reporter

# Foundation depth below the lowest ground point under a footprint.
EMBED_M = 0.1


def build_mesh(value, state):
    bbox = state.require_metadata("bbox")
    data = extrude_buildings(
        value["osm"],
        bbox,
        reporter=get_reporter(state),
        terrain=state.get_metadata("terrain", None),
    )
    state.require_metadata("progress_monitor").next()
    return data


def ground_under(corners, terrain) -> tuple[float, float]:
    """Lowest and highest ground (m above terrain datum) under a footprint."""
    pts = np.asarray(corners, dtype=float)[:, :2]
    poly = Polygon(pts)
    if poly.is_valid and poly.area > 0:
        step = terrain.spacing
        ring = np.asarray(poly.exterior.segmentize(step).coords)[:, :2]
        minx, miny, maxx, maxy = poly.bounds
        gx, gy = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
        inside = np.column_stack([gx.ravel(), gy.ravel()])
        inside = inside[shapely.contains_xy(poly, inside[:, 0], inside[:, 1])]
        pts = np.vstack([ring, inside])
    ground = terrain.ground(pts[:, 0], pts[:, 1])
    return float(ground.min()), float(ground.max())

def run_extrusion(id, buildings, element, area_bbox, terrain):
    corners = get_corners(element, area_bbox)
    
    lines = get_lines(corners)

    path = trimesh.path.path.Path2D(
        entities=lines,
        vertices=corners,
    )

    base, top = get_base_and_height(element)

    polys = path.polygons_closed
    if not polys[0]:
        return

    # Meshes here are z-down (texturing and export flip z). On terrain the
    # roof sits `top` above the highest ground under the footprint and
    # the walls reach down to the lowest, so nothing floats downhill. A
    # building part with a base (min_height: e.g. a tower's upper section)
    # starts that high above the ground instead.
    bottom = base
    height = top - base
    if terrain is not None:
        ground_low, ground_high = ground_under(corners, terrain)
        bottom = ground_high + base if base > 0 else ground_low - EMBED_M
        height = ground_high + top - bottom

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

    if bottom:
        mesh.apply_translation([0.0, 0.0, -bottom])

    if id in buildings:
        buildings[id] = trimesh.util.concatenate([buildings[id], mesh])
    else:
        buildings[id] = mesh

def extrude_buildings(
    input_data, area_bbox, reporter: Reporter | None = None, terrain=None
):
    reporter = reporter or get_reporter()
    buildings = {}

    for _, element in reporter.progress(
        enumerate(input_data["elements"]),
        desc="Generating Building Mesh(s)",
        total=len(input_data["elements"]),
    ):
        id = element["id"]

        # some buildings like the Dude are not represented as a "way" type, they are represented as a "relation" type, so it was failing the first check
        # I have split up the checks, but now need to handle the extrusion different since the types store things differently
        if element["type"] == "way":
            if "geometry" not in element:
                continue

        if element["type"] == "relation":
            for member in element["members"]:
                if "geometry" not in member:
                    continue

        if element["type"] == "way":
            run_extrusion(id, buildings, element, area_bbox, terrain)
        elif element["type"] == "relation":
            for member in element["members"]:
                run_extrusion(id, buildings, member, area_bbox, terrain)

    combined_mesh = trimesh.util.concatenate(
        [building for _, building in buildings.items()]
    )
    return combined_mesh
