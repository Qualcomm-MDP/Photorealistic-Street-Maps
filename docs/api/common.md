# Utilities API

Common utility modules used across the pipeline for coordinate conversion, mesh operations, progress tracking, and segmentation.

## Mesh Builder

### extrude_buildings

The core mesh generation function that converts OSM building data to 3D meshes.

::: mesh_builder.extrude.extrude_buildings
    options:
      show_source: true

::: mesh_builder.extrude.build_mesh
    options:
      show_source: true

## Texturing

### apply_textures

Projects street-view images onto mesh faces using camera geometry.

::: texturing.tex_projection.apply_textures
    options:
      show_source: true

## Segmentation

### Obstruction Removal

The segmentation module uses Mask2Former for semantic segmentation and LaMa for inpainting to remove obstructions from street-view images.

::: segmentation.obstruction.remove_obstructions
    options:
      show_source: true

::: segmentation.obstruction.build_obstruction_mask
    options:
      show_source: true

::: segmentation.obstruction.synthesize_texture
    options:
      show_source: true

### Obstruction Classes

The following Cityscapes semantic class IDs are treated as obstructions and masked for inpainting:

| Class ID | Category |
|---|---|
| 5 | Pole |
| 6 | Traffic light |
| 7 | Traffic sign |
| 8 | Vegetation |
| 11 | Car |
| 12 | Truck |

## Mesh Utilities

Geometry helper functions for coordinate conversion and building parameter extraction.

::: common.MeshUtils
    options:
      show_source: true

## Mesh Export

GLB export functionality.

::: common.MeshExport
    options:
      show_source: true

## Coordinate Conversions

Utilities for converting between coordinate systems (WGS84, UTM, pixel).

::: common.conversions.basic
    options:
      show_source: true

::: common.conversions.utm
    options:
      show_source: true

## ProgressMonitor

CLI progress display for tracking pipeline stages.

::: common.ProgressMonitor
    options:
      show_source: true
