# Terrain & Aerial Photos

Two optional stages add the ground under the buildings. Both are off by default; without them the output is a set of buildings on flat ground with grey roofs.

| Turn on                      | Command line | Web app                     |
| ---------------------------- | ------------ | --------------------------- |
| Real ground heights          | `--terrain`  | **Terrain** option          |
| Aerial photos (ground, roofs) | `--naip`    | **Aerial photos** option    |

Aerial photos are draped on the terrain, so `--naip` turns `--terrain` on too.

Neither can break a run: if a download fails, the stage logs a warning and the pipeline continues with flat ground, or with plain ground and grey roofs.

## Terrain (`terrain`)

`src/terrain/terrain.py` builds a grid of ground heights for the area.

1. **Elevation tiles**: it downloads [Terrarium](https://github.com/tilezen/joerd/blob/master/docs/formats.md#terrarium) elevation tiles (AWS Open Data) at zoom 15, about 3.5 m per pixel at Ann Arbor's latitude, and decodes the heights from their colours.
2. **Grid**: it samples them onto a regular **2 m** grid in the area's UTM zone. The grid reaches **100 m** past the bounding box, because OpenStreetMap returns whole buildings that cross the box's edge and their ground would otherwise be extrapolated.
3. The grid is stored in the state as `terrain`. Heights are in metres above the lowest point of the grid.

The terrain is used three times:

- **Buildings** (build mesh): each building's walls reach **0.1 m below the lowest ground** under its footprint, and its roof sits its height above the **highest ground** under the footprint. On a slope nothing floats and nothing is buried.
- **Cameras** (texture): each street photo's camera sits **1.6 m above the ground** at its position, instead of 1.6 m above a flat zero.
- **Ground** (export): the grid becomes a ground mesh, exported as its own `terrain` object next to the `buildings`.

## Aerial photos (`naip`)

`src/naip/naip.py` textures the ground and the roofs with [NAIP](https://naip-usdaonline.hub.arcgis.com/) (USDA National Agriculture Imagery Program) photos: public-domain, north-up, about **0.6 m per pixel**, covering the continental US.

1. **Download**: it requests the newest natural-colour NAIP image over the terrain grid from the USGS ImageServer, directly in the area's UTM zone, so pixels map linearly to metres. Large areas are fetched in tiles of up to 4000 × 4000 pixels.
2. **Roof alignment**: OSM footprints and the roofs in the photo are usually a few metres apart (georegistration error, and buildings leaning away from straight down). One shift for the whole area is found by matching the footprint outlines against edges in the photo, with an FFT cross-correlation that scores every shift up to **±15 m** at once. If the best shift lands at the edge of that range, it's probably spurious: no shift is applied and a warning is logged.
3. **Roofs**: up-facing faces get their texture cut from the photo at the shifted position, and those patches go into the same texture atlas as the street photos. The buildings stay on the OSM footprints and remain one mesh with one material. Roof faces are no longer given to street photos.
4. **Ground**: the roofs (at both the footprint and the shifted position, grown by 1.2 m) are painted out of the photo with OpenCV inpainting, so no "ghost roof" appears on the ground next to a building. The export drapes the result on the terrain mesh from above.

:::note[Coverage]

NAIP covers the continental US only; outside it there's no imagery to use, so don't turn aerial photos on there. The web app shows a note when the selected area looks outside the continental US.

:::

## Source code

- `src/terrain/terrain.py`: `build_terrain()`, `Terrain`, the `terrain` stage
- `src/naip/naip.py`: `fetch_naip()`, `estimate_roof_offset()`, `build_ortho()`, the `naip` stage
- `src/mesh_builder/extrude.py`: buildings on the terrain
- `src/texturing/tex_projection.py`: camera heights and roof patches
- `src/pipeline.py`: the ground in the export
