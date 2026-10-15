"""
blender_render_views.py - batch-render a GLB from a list of camera poses.

Called by render_mapillary_views.py; not meant to be run by hand:
    blender -b --python blender_render_views.py -- model.glb jobs.json

jobs.json is a list of {"out", "width", "height", "matrix" (4x4 camera-to-world),
"lens_mm", "grey"}. Render settings match setup_render.py: Workbench, flat
lighting, texture colours, "Standard" view transform, transparent background.
"""
import json
import sys

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
glb_path, jobs_path = argv[0], argv[1]
jobs = json.load(open(jobs_path))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=glb_path)

cam_data = bpy.data.cameras.new("EvalCamera")
cam_data.sensor_fit = "AUTO"
cam = bpy.data.objects.new("EvalCamera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.single_color = (0.5, 0.5, 0.5)
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.resolution_percentage = 100

for job in jobs:
    scene.render.resolution_x = job["width"]
    scene.render.resolution_y = job["height"]
    scene.display.shading.color_type = "SINGLE" if job.get("grey") else "TEXTURE"
    cam.matrix_world = mathutils.Matrix(job["matrix"])
    cam_data.lens = job["lens_mm"]
    scene.render.filepath = job["out"]
    bpy.ops.render.render(write_still=True)
