# Photorealistic Open Street Map (POSM)

**An open-source pipeline for generating photorealistic 3D street-level maps from GPS coordinates alone.**

---

## What is POSM?

POSM is an end-to-end pipeline that takes a geographic bounding box as input and produces a textured 3D mesh (`.glb` file) of the urban environment within that region. It combines open geospatial data from OpenStreetMap and satellite/street-view imagery from NAIP and Mapillary to generate simulation-grade 3D cityscapes — without proprietary datasets, LiDAR hardware, or costly API access.

This project is developed as part of the **Qualcomm Multidisciplinary Design Program (MDP)** at the University of Michigan.

## The Problem

There is currently no open-source, automated pipeline for producing photorealistic 3D street-level maps from GPS coordinates. Existing solutions such as Google Maps 3D are proprietary, hardware-dependent, time-intensive, and unscalable. This creates a gap for anyone building systems that need realistic environment representations, including robotics and drone navigation, natural disaster simulation, autonomous vehicles, and digital twin applications.

## How It Works

The pipeline runs in four stages:

1. **Data Processing** — Fetches building footprints from OSM, street-view images from Mapillary, and aerial imagery from USGS NAIP. Filters and scores images per building using Mask2Former segmentation.
2. **Mesh Generation** — Extrudes OSM building footprints into 3D geometry using height data, producing a combined trimesh scene.
3. **Texturing** — Projects street-view images onto mesh faces via camera projection, removes obstructions with LaMa inpainting, and assembles a texture atlas.
4. **Evaluation** — Measures pipeline runtime, resource usage, and output quality through profiling and user surveys.

## Quick Example

```bash
poetry run python src/main.py
```

The pipeline will prompt you for bounding box coordinates (south, west, north, east), then produce a textured `.glb` file you can open in any 3D viewer.

## Who Benefits

| Audience                | Use Case                                 |
| ----------------------- | ---------------------------------------- |
| Robotics & AV teams     | Realistic simulation environments        |
| Emergency response      | Flood/disaster digital twin modeling     |
| Insurance & quant firms | Geospatial risk assessment               |
| Open-source community   | Foundation for 3D urban generation tools |
| Qualcomm (internal)     | Wireless R&D, digital twin applications  |

## Project Links

- **Repository:** [github.com/Qualcomm-MDP/Photorealistic-Street-Maps](https://github.com/Qualcomm-MDP/Photorealistic-Street-Maps)
- **Design Review 1 Slides:** See the `docs/assets/` folder in the repository

---

*Built at the University of Michigan in partnership with Qualcomm.*
