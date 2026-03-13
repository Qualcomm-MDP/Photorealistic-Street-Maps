import trimesh
from common.MeshUtils import initialize_plane, get_corners, get_lines, get_height
from common.constants import BoundingBox


def extrude_buildings(input_data, area_bbox):
    # Just helps for debugging
    # trimesh.util.attach_to_log()
    scale = 5

    _, plane_vertices, plane_faces = initialize_plane(area_bbox, scale)
    mesh = trimesh.Trimesh(
        vertices=plane_vertices,
        faces=plane_faces,
    )

    buildings = []
    buildings.append(mesh)

    for _, element in enumerate(input_data["elements"]):
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

        height = -1 * height
        mesh = path.extrude(height=height)

        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate([
                m.to_mesh() if hasattr(m, "to_mesh") else m
                for m in mesh
            ])
        else:
            if hasattr(mesh, "to_mesh"):
                mesh = mesh.to_mesh()

        # # Apply the wrap to the mesh
        # texture_img = PIL.Image.open(WRAP_IMG)
        # textured_mesh = mesh.unwrap(texture_img)
        # buildings.append(textured_mesh)

        # Export it out in glb format
        
        # Add that to the list of all the elements
        buildings.append(mesh)
    
    # Combine the meshes into one just so that we can use it as a singular mesh
    combined_mesh = trimesh.util.concatenate(buildings)
    print("Saved combined building mesh!\n")
    combined_mesh.export(f"combined.glb", file_type='glb')