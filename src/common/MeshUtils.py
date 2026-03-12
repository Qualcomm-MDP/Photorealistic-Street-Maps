import trimesh

# Follow https://github.com/Qualcomm-MDP/mesh_creation/blob/main/extrude_out.py
def generate_plane(height, width):
    # Need to generate the vertexes that are in the perimeter

    # Just make the corners to the size of the bounding box
    corners = [
        [0, 0, 0],
        [0, width, 0],
        [height, width, 0],
        [height, 0, 0],
    ]

    # Tells trimesh that I want a face represented by corners[0], corners[1], corners[2], and corners[3]
    faces = np.array([[0, 1, 2, 3]])

    # Generate the plane :)
    plane = trimesh.Trimesh(vertices=corners, faces=faces)

    return plane, corners, faces

def initialize_plane(data):
    # Rounding to a decimal place of 5 gives us around 1 m accuracy in the real world (worst case at equator), according to Chat
    max_lat = int(MAX_LAT * (10**SCALE))
    min_lat = int(MIN_LAT * (10**SCALE))
    max_lon = int(MAX_LON * (10**SCALE))
    min_lon = int(MIN_LON * (10**SCALE))

    delta_lat = abs(max_lat - min_lat) # width of our underlying plane mesh
    delta_long = abs(max_lon - min_lon) # height of our underlying plane mesh

    plane, corners, faces = generate_plane(delta_lat, delta_long) # plane is a mesh
    return plane, corners, faces

def get_corners(element):
    # Go into the API to get the geometry lat, lon points (representative of the nodes)
    geometry_points = element["geometry"]
    corners = []

    # Go through each geometry point, and take the lat and lon and convert it to x and y
    for point in geometry_points:
        # Convert the latitude and longitude into ints (multiplying by 100,000 to get rid of decimals)
        latitude = int(float(point["lat"]) * (10**SCALE))
        longitude = int(float(point["lon"]) * (10**SCALE))

        # Calculate the offset relative to the min and max bounds of the region
        local_i = abs(latitude - int(MIN_LAT * (10**SCALE)))
        local_j = abs(longitude - int(MIN_LON * (10**SCALE)))

        # Add the coordinates to the corners
        corners.append([local_i, local_j])
    
    return corners

# Get the trimesh lines that connects the corners together
def get_lines(corners, loop=True):
    lines = []
    # Lines go from corners[start] to corners[end]
    # Since Overpass already has the geometry nodes in order, we just need to loop from 0-1, 1-2, etc and then [end, 0] to wrap back to beginning, close the polygon
    start = 0
    end = 1
    lines.append(trimesh.path.entities.Line([start, end])) # Add in the start of the path

    # Go until we hit the end and add in those lines
    for i in range(len(corners) - 2):
        start += 1
        end += 1
        lines.append(trimesh.path.entities.Line([start, end]))

    # Add in that last trimesh line that puts us back at the beginning (closes the polygon)
    if loop:
        lines.append(trimesh.path.entities.Line([end, 0]))
    return lines

def get_height():
    def get_height(element):
    # Some elements have a height attribute in their tags section
    height = element["tags"].get("height") # try to get the height value
    if not height:
        # If there is no height, get the building levels tag
        height = element["tags"].get("building:levels")

        if not height:
            # If there isn't any sort of building leves, just set the height to be one story, times the height of one story (3 meters?)
            height = float(3 * 1)
        else:
            # Otherwise, set the height to be the number of stories * 3 meters
            height = 3 * float(height)
    else:
        height = float(height)
    
    return height
