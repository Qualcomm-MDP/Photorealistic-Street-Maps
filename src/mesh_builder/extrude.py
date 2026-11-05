import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon

from common.MeshUtils import get_corners, get_lines, get_height
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


def extrude_buildings(
    input_data, area_bbox, reporter: Reporter | None = None, terrain=None
):
    reporter = reporter or get_reporter()
    buildings = {}
    flagged = []  # (osm, reason for buildings left out of mesh)

    for _, element in reporter.progress(
        enumerate(input_data["elements"]),
        desc="Generating Building Mesh(s)",
        total=len(input_data["elements"]),
    ):
        osm = f"{element.get('type')} {element.get('id')}"
        if "geometry" not in element:
            flagged.append((osm, "no outline in data"))
            continue

        id = element["id"]
        corners = get_corners(element, area_bbox)

        lines = get_lines(corners)

        path = trimesh.path.path.Path2D(
            entities=lines,
            vertices=corners,
        )

        try:
            height = get_height(element)
        except KeyError:
            flagged.append((osm, "no tags"))
            continue
        except ValueError as exc:
            flagged.append((osm, f"unreadable height tag: {exc}"))
            continue

        polys = path.polygons_closed
        if len(polys) == 0 or not polys[0]:
            flagged.append((osm, "outline is not a valid closed shape"))
            continue

        # Meshes here are z-down (texturing and export flip z). On terrain the
        # roof sits `height` above the highest ground under the footprint and
        # the walls reach down to the lowest, so nothing floats downhill.
        bottom = 0.0
        if terrain is not None:
            ground_low, ground_high = ground_under(corners, terrain)
            bottom = ground_low - EMBED_M
            height = ground_high + height - bottom

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

        buildings[id] = mesh

    if flagged:
        details = "\n".join(f"  {osm}: {reason}" for osm, reason in flagged)
        reporter.warning(
            f"{len(flagged)} building(s) were left out of the mesh:\n{details}"
        )

    combined_mesh = trimesh.util.concatenate(
        [building for _, building in buildings.items()]
    )
    return combined_mesh
