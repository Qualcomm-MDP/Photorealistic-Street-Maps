import shapely
from shapely.geometry import Polygon
from pyproj import Transformer
import trimesh
import numpy as np
import json
import PIL
import matplotlib.pyplot as plt

from common.MeshUtils import generate_plane, initialize_plane, get_corners, get_lines, get_height

# Global, path to the outputted JSON file
INPUT_JSON = "osm_data_buildings.json" 
STREET_JSON = "osm_data_roads.json"
SCALE = 5 # What level of precision we want

WRAP_IMG = "wraps/birdseye.png"

def main():
    # Just helps for debugging
    # trimesh.util.attach_to_log()
    data_buildings = None
    
    # Read in the json data
    with open(INPUT_JSON, "r") as f:
        data_buildings = json.load(f)

    # Declare the bounds as globals so that everyone can use them
    global MAX_LAT
    global MIN_LAT
    global MAX_LON
    global MIN_LON

    # Set the global bounds
    MAX_LAT = float(data_buildings["max_lat"])
    MIN_LAT = float(data_buildings["min_lat"])
    MAX_LON = float(data_buildings["max_lon"]) 
    MIN_LON = float(data_buildings["min_lon"])
    
    # Initialize that initial plane, great name I know
    hisodflkjas, plane_vertices, plane_faces = initialize_plane(data_buildings)
    st_img = PIL.Image.open(WRAP_IMG)

    # Add the satellite image of the plane in order to have some sort of a street view
    uv = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    material = trimesh.visual.texture.SimpleMaterial(image=st_img)
    print(material)

    texture = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=st_img
    )

    mesh = trimesh.Trimesh(
        vertices=plane_vertices,
        faces=plane_faces,
        visual=texture
    )

    # buildings will be the combined mesh of everything in the scene, so the underlying plane, buildings, roads etc.
    buildings = []
    buildings.append(mesh) # Add in the plane to begin

    # Go through the elements returned by overpass' API
    for i, element in enumerate(data_buildings["elements"]):

        id = element["id"]
        # Get the corners
        corners = get_corners(element)

        # Get the lines
        lines = get_lines(corners)

        # Create the 2D Path for the mesh
        path = trimesh.path.path.Path2D(
            entities=lines,
            vertices=corners,
        )

        # Get the height of the buildings
        height = get_height(element)

        # Extra check to make sure that the outline of the building is a valid one
        polys = path.polygons_closed
        if not polys[0]:
            continue # Don't try to add a mesh if it is going to be invalid

        # Get the mesh for that building
        height = -1 * height
        mesh = path.extrude(height=height)
        print(type(mesh))
        mesh = path.extrude(height=height)

        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate([
                m.to_mesh() if hasattr(m, "to_mesh") else m
                for m in mesh
            ])
        else:
            if hasattr(mesh, "to_mesh"):
                mesh = mesh.to_mesh()

        print("Saved building mesh!\n")
        mesh.export(f"output_meshes/{id}.glb", file_type='glb')

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
    combined_mesh.export(f"combined.glb")

if __name__ == "__main__":
    main()