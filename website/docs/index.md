---
slug: /
title: PSM
sidebar_label: Home
---

# Photorealistic Street Maps (PSM)

**An open-source pipeline for generating photorealistic 3D street-level maps from GPS coordinates alone.**

## What is PSM?

PSM takes a geographic bounding box and produces a textured 3D model (`.glb`) of the buildings in it. It uses open data only: building footprints from OpenStreetMap, street-level photos from Mapillary, and, optionally, ground heights from Terrarium elevation tiles and USDA NAIP aerial photos. No proprietary datasets, LiDAR hardware, or paid APIs are needed.

You can run it from a **web app** (pick an area on a map, watch the job, preview and download the model) or from the **command line**.

This project is developed as part of the **Qualcomm Multidisciplinary Design Program (MDP)** at the University of Michigan.

## The Problem

There is currently no open-source, automated pipeline for producing photorealistic 3D street-level maps from GPS coordinates. Existing solutions such as Google Maps 3D are proprietary, hardware-dependent, time-intensive, and unscalable. This creates a gap for anyone building systems that need realistic environment representations, including robotics and drone navigation, natural disaster simulation, autonomous vehicles, and digital twin applications.

## How It Works

The pipeline runs these stages in order. The optional ones do nothing unless they're turned on.

1. **Fetch data**: building footprints from OpenStreetMap and street photos (with camera poses) from Mapillary.
2. **Terrain** *(optional)*: real ground heights for the area.
3. **Build mesh**: extrudes each footprint into a 3D building, placed on the terrain when there is one.
4. **Aerial photo** *(optional)*: NAIP aerial imagery for the ground and roofs.
5. **Texture**: removes cars, people and poles from the street photos (Mask2Former + LaMa) and projects them onto the building walls.
6. **Export**: writes the `.glb`.

See [Pipeline Overview](pipeline/overview.md) for the details of each stage.

## Quick Start

The quickest way is the web app in Docker:

```bash
docker compose up --build api
```

Then open <http://localhost:8000>, add your Mapillary token in **Settings**, pick an area and press **Generate model**. See [Getting Started](getting-started.mdx) for the other ways to install and run PSM.

## Who Benefits

| Audience                | Use Case                                 |
| ----------------------- | ---------------------------------------- |
| Robotics & AV teams     | Realistic simulation environments        |
| Emergency response      | Flood/disaster digital twin modeling     |
| Insurance & quant firms | Geospatial risk assessment               |
| Open-source community   | Foundation for 3D urban generation tools |
| Qualcomm (internal)     | Wireless R&D, digital twin applications  |

## Project Links

- **Repository:** [github.com/Qualcomm-MDP/PSM](https://github.com/Qualcomm-MDP/PSM)
