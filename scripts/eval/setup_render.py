"""
setup_render.py - turn the pipeline's .glb into an image you can compare to a real photo.

Run from a terminal (Blender must be installed):
    blender --python setup_render.py -- model.glb heldout_photo.jpg

On a Mac, "blender" is usually:
    /Applications/Blender.app/Contents/MacOS/Blender --python setup_render.py -- model.glb heldout_photo.jpg

Blender opens with:
  - your model loaded
  - a camera whose output size matches the photo exactly
  - the real photo shown semi-transparently in the camera view, so you can line things up
  - render settings for evaluation: textures shown exactly as-is (no extra lighting
    or shadows) and a transparent background

Then follow the steps printed in the terminal (also listed at the bottom of this file).
"""
import sys

import bpy
import mathutils

# ---- read the two file paths given after "--" -------------------------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    raise SystemExit("usage: blender --python setup_render.py -- model.glb photo.jpg [X Y [HEADING_DEG]]")
glb_path, photo_path = argv[0], argv[1]
# NOTE: Blender ignores the rotation flag phones store in photos. If the photo
# appears sideways in the camera view, save an upright copy first (e.g. open and
# re-export it in Preview) and use that copy for BOTH Blender and simple_compare.

# ---- start from an empty scene ----------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---- load the model ----------------------------------------------------------
bpy.ops.import_scene.gltf(filepath=glb_path)

# The export centres each model vertically, so the ground sits at minus half the
# building's height (z = -3 for a 6 m building). Measure it rather than assume it.
meshes = [o for o in scene.objects if o.type == "MESH"]
ground_z = min((o.matrix_world @ mathutils.Vector(c)).z for o in meshes for c in o.bound_box)
eye_z = ground_z + 1.6

# ---- load the photo and match the render size to it --------------------------
photo = bpy.data.images.load(photo_path)
w, h = photo.size
scene.render.resolution_x = w
scene.render.resolution_y = h
scene.render.resolution_percentage = 100

# ---- camera, with the photo overlaid in its view -----------------------------
cam_data = bpy.data.cameras.new("EvalCamera")
cam_data.sensor_fit = "AUTO"
cam_data.lens = 26  # typical phone main camera (35mm-equivalent); adjust if the photo looks zoomed
cam_data.show_background_images = True
bg = cam_data.background_images.new()
bg.image = photo
bg.alpha = 0.5
bg.display_depth = "FRONT"   # draw the photo over the model so you can see both
bg.frame_method = "FIT"
cam = bpy.data.objects.new("EvalCamera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
# Axes after import: X = east, Y = north, Z = up (metres).
cam.location = (0.0, -30.0, eye_z)
cam.rotation_euler = (1.5708, 0.0, 0.0)  # looking horizontally north (+Y)

# Optional exact placement:  ... -- model.glb photo.jpg X Y [HEADING_DEG]
#   X = UTM easting  - glb_to_utm[0][3]   (from the sidecar JSON; these offsets
#   Y = UTM northing - glb_to_utm[1][3]    differ for every export)
#   HEADING_DEG = compass direction the photo faces (0 = north, 90 = east),
#   e.g. from the photo's EXIF GPSImgDirection if the phone recorded it.
if len(argv) >= 4:
    cam.location = (float(argv[2]), float(argv[3]), eye_z)
if len(argv) >= 5:
    import math
    cam.rotation_euler = (math.radians(90), 0.0, -math.radians(float(argv[4])))

# ---- evaluation render settings ---------------------------------------------
# Workbench + flat lighting shows the textures exactly as the pipeline made them,
# so the score reflects the texture, not Blender's lighting choices.
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "TEXTURE"
scene.render.film_transparent = True          # background becomes transparent
# Blender 4+/5 default "AgX" color management shifts colors (~9/255 on our test);
# "Standard" saves the texture colors exactly as the pipeline made them.
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"  # keep the transparency when saving

# ---- show the camera view in the 3D viewport --------------------------------
for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.region_3d.view_perspective = "CAMERA"
                    space.lock_camera = True        # moving the view moves the camera
                    # Workbench has no Material Preview; Rendered shows exactly what F12 renders
                    space.shading.type = "RENDERED"

print("""
=====================================================================
 READY. To line up and save the render:

 1. Hover over the 3D view. You're looking through the camera, with the
    real photo drawn semi-transparently on top.
 2. Move the view until the building lines up with the photo:
      middle-mouse drag      = rotate
      Shift + middle drag    = pan
      scroll                 = move forward/back
    ("Lock Camera to View" is on, so the camera follows.)
    Get the building's corners and roofline to match. Ignore trees and cars.
 3. If the building looks too big or small no matter where you move,
    change the focal length: select EvalCamera > Camera properties > Focal Length.
 4. Press F12 to render.
 5. In the render window: Image > Save As > PNG. Make sure the color mode is RGBA.
    Name it after the photo, e.g. dorm_photo3_render.png
 6. File > Save As to keep this .blend, so you can re-render later
    without re-aligning (e.g. after the pipeline improves).
=====================================================================
""")
