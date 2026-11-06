"""NAIP aerial photos for the ground and the roofs.

NAIP (USDA National Agriculture Imagery Program) is public-domain, ~0.6 m/px,
north-up imagery of the continental US. The USGS ImageServer reprojects it into
our UTM zone on request, so pixel (row, col) maps linearly to UTM (x, y).

Roofs in NAIP rarely sit exactly on the OSM footprints (georegistration error
and building lean away from nadir), so one shift for the whole area is found by
matching footprint outlines to image edges. Buildings stay on the OSM
footprints; only the roof texture is sampled at the shifted position.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from trimesh.visual.material import PBRMaterial

from common import PipelineState
from common.MeshUtils import _get_utm_transformer
from common.reporter import Reporter, get_reporter

NAIP_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPImagery/ImageServer/exportImage"
)
NAIP_RES_M = 0.6
NAIP_MAX_PX = 4000  # service limit per request; larger areas are tiled
# Roof faces point down in build_mesh's z-down frame.
ROOF_MAX_NZ = -0.9
ROOF_SEARCH_M = 15.0
# Extra ground removed around each roof (eaves, walls, shadow edges).
INPAINT_GROW_M = 1.2
# Real neighbouring pixels kept around each roof patch so texture filtering
# never blends in the next patch of the atlas.
PATCH_PAD_PX = 2


@dataclass
class Ortho:
    """North-up aerial photo on the UTM grid.

    image is RGB uint8 with row 0 at the north edge and column 0 at the west
    edge; each pixel is res metres square. ground is the same photo with the
    roofs painted out. A roof at UTM (x, y) appears in image at
    (x + roof_dx, y + roof_dy).
    """

    image: np.ndarray
    west: float
    north: float
    res: float
    roof_dx: float = 0.0
    roof_dy: float = 0.0
    ground: np.ndarray | None = field(default=None, repr=False)

    @property
    def width_m(self) -> float:
        return self.image.shape[1] * self.res

    @property
    def height_m(self) -> float:
        return self.image.shape[0] * self.res

    def to_px(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """UTM -> continuous (col, row); pixel edges are at whole numbers."""
        return (np.asarray(x) - self.west) / self.res, (
            self.north - np.asarray(y)
        ) / self.res

    def drape(self, vertices: np.ndarray) -> trimesh.visual.TextureVisuals:
        """Ground texture seen from straight above, for z-up UTM vertices."""
        u = (vertices[:, 0] - self.west) / self.width_m
        v = 1.0 - (self.north - vertices[:, 1]) / self.height_m
        ground = self.image if self.ground is None else self.ground
        material = PBRMaterial(
            name="naip_ground",
            baseColorTexture=Image.fromarray(ground),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        return trimesh.visual.TextureVisuals(
            uv=np.clip(np.column_stack([u, v]), 0.0, 1.0), material=material
        )


def fetch_naip(
    west: float,
    south: float,
    east: float,
    north: float,
    epsg: int,
    res: float = NAIP_RES_M,
    reporter: Reporter | None = None,
) -> Ortho:
    """Newest NAIP natural-colour image covering the UTM box (may extend it by < 1 px)."""
    reporter = reporter or get_reporter()
    w_px = int(np.ceil((east - west) / res))
    h_px = int(np.ceil((north - south) / res))
    image = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    params_base: dict[str, str | int] = {
        "bboxSR": epsg,
        "imageSR": epsg,
        "format": "jpgpng",
        "pixelType": "U8",
        "interpolation": "RSP_BilinearInterpolation",
        "renderingRule": json.dumps({"rasterFunction": "NaturalColor"}),
        "f": "image",
    }
    session = requests.Session()
    n_requests = 0
    for r0 in range(0, h_px, NAIP_MAX_PX):
        for c0 in range(0, w_px, NAIP_MAX_PX):
            w, h = min(NAIP_MAX_PX, w_px - c0), min(NAIP_MAX_PX, h_px - r0)
            x0, y1 = west + c0 * res, north - r0 * res
            params = dict(
                params_base,
                bbox=f"{x0},{y1 - h * res},{x0 + w * res},{y1}",
                size=f"{w},{h}",
            )
            r = session.get(NAIP_URL, params=params, timeout=180)
            r.raise_for_status()
            if not r.headers.get("Content-Type", "").startswith("image"):
                raise RuntimeError(f"NAIP service returned non-image: {r.text[:200]}")
            tile = Image.open(io.BytesIO(r.content)).convert("RGB")
            if tile.size != (w, h):
                tile = tile.resize((w, h), Image.BILINEAR)
            image[r0 : r0 + h, c0 : c0 + w] = np.asarray(tile)
            n_requests += 1
    reporter.info(
        f"  NAIP image {w_px}x{h_px} px at {res:g} m/px ({n_requests} request(s))"
    )
    return Ortho(image=image, west=west, north=north, res=res)


def roof_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Indices of the up-facing (roof) faces of a build_mesh (z-down) mesh."""
    return np.nonzero(mesh.face_normals[:, 2] < ROOF_MAX_NZ)[0]


def roof_mask(
    mesh: trimesh.Trimesh, ortho: Ortho, dx: float = 0.0, dy: float = 0.0
) -> np.ndarray:
    """Where the roofs, shifted by (dx, dy) metres, cover the image grid."""
    mask = np.zeros(ortho.image.shape[:2], dtype=np.uint8)
    tris = mesh.triangles[roof_faces(mesh)]
    cols, rows = ortho.to_px(tris[..., 0] + dx, tris[..., 1] + dy)
    # cv2 puts integer coordinates on pixel centres; 3 fractional bits.
    pts = np.round(np.stack([cols - 0.5, rows - 0.5], axis=-1) * 8).astype(np.int32)
    for tri in pts:
        cv2.fillPoly(mask, [tri], 1, lineType=cv2.LINE_8, shift=3)
    return mask.astype(bool)


def _image_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|d/dx| and |d/dy| of smoothed luminance, scaled to ~0..1, mean removed."""
    grey = gaussian_filter(image.astype(np.float32).mean(axis=2), 1.0)
    ex, ey = np.abs(sobel(grey, axis=1)), np.abs(sobel(grey, axis=0))
    scale = float(np.percentile(np.hypot(ex, ey), 99)) + 1e-6
    ex, ey = np.clip(ex / scale, 0, 1), np.clip(ey / scale, 0, 1)
    return ex - ex.mean(), ey - ey.mean()


def _parabolic_peak(W: np.ndarray) -> tuple[float, float, float, bool]:
    """Argmax of a score window with sub-pixel refinement; flags edge peaks."""
    i, j = np.unravel_index(int(np.argmax(W)), W.shape)
    on_edge = i in (0, W.shape[0] - 1) or j in (0, W.shape[1] - 1)

    def vertex(a, b, c):
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if den < 0 else 0.0

    di = 0.0 if on_edge else vertex(W[i - 1, j], W[i, j], W[i + 1, j])
    dj = 0.0 if on_edge else vertex(W[i, j - 1], W[i, j], W[i, j + 1])
    return i + di, j + dj, float(W[i, j]), on_edge


def estimate_roof_offset(
    image: np.ndarray,
    mask: np.ndarray,
    res: float,
    search_m: float = ROOF_SEARCH_M,
    reporter: Reporter | None = None,
) -> tuple[float, float]:
    """Shift (dx, dy) in metres east/north that best lines roof outlines up with image edges.

    Scores every shift up to +-search_m at once with an FFT cross-correlation
    of x-edges with x-edges and y-edges with y-edges. A best shift at the edge
    of the search window is probably spurious, so (0, 0) is returned then.
    """
    reporter = reporter or get_reporter()
    if not mask.any():
        return 0.0, 0.0
    ex, ey = _image_edges(image)
    m = mask.astype(np.float32)
    mx, my = np.abs(sobel(m, axis=1)) / 4.0, np.abs(sobel(m, axis=0)) / 4.0
    R = max(1, int(np.ceil(search_m / res)))
    shape = (ex.shape[0] + R, ex.shape[1] + R)  # zero padding: no wrap within +-R
    F = np.fft.rfft2
    corr = np.fft.irfft2(
        F(ex, shape) * np.conj(F(mx, shape)) + F(ey, shape) * np.conj(F(my, shape)),
        shape,
    )
    win = np.roll(corr, (R, R), axis=(0, 1))[: 2 * R + 1, : 2 * R + 1]
    win /= max(float(mx.sum() + my.sum()), 1e-6)
    i, j, peak, on_edge = _parabolic_peak(win)
    dx, dy = (j - R) * res, -(i - R) * res
    if on_edge:
        reporter.warning(
            f"Roof alignment peaked at the +-{search_m:g} m search limit "
            f"(dx={dx:+.1f} m, dy={dy:+.1f} m), probably spurious; using no shift"
        )
        return 0.0, 0.0
    reporter.info(
        f"  Roof offset dx={dx:+.2f} m east, dy={dy:+.2f} m north "
        f"(edge match {peak:.3f} vs {float(win[R, R]):.3f} unshifted)"
    )
    return dx, dy


def remove_roofs(image: np.ndarray, mask: np.ndarray, res: float) -> np.ndarray:
    """Paint the masked roofs (grown by INPAINT_GROW_M) out of the ground photo."""
    grow = max(1, int(round(INPAINT_GROW_M / res)))
    kernel = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    grown = cv2.dilate(mask.astype(np.uint8), kernel)
    return cv2.inpaint(np.ascontiguousarray(image), grown, 8, cv2.INPAINT_TELEA)


def roof_patches(
    verts_up: np.ndarray, faces: np.ndarray, roof_idx: np.ndarray, ortho: Ortho
) -> list[dict]:
    """Atlas patch records (as in tex_projection) cropping each roof face from NAIP."""
    records = []
    h, w = ortho.image.shape[:2]
    for fi in roof_idx:
        tri = verts_up[faces[fi]]
        cols, rows = ortho.to_px(tri[:, 0] + ortho.roof_dx, tri[:, 1] + ortho.roof_dy)
        c0 = int(np.floor(cols.min())) - PATCH_PAD_PX
        r0 = int(np.floor(rows.min())) - PATCH_PAD_PX
        c1 = int(np.ceil(cols.max())) + PATCH_PAD_PX
        r1 = int(np.ceil(rows.max())) + PATCH_PAD_PX
        if c0 < 0 or r0 < 0 or c1 > w or r1 > h:
            continue

        patch_rgb = ortho.image[r0:r1, c0:c1]
        # Add fully opaque alpha channel
        alpha = np.full(patch_rgb.shape[:2], 255, dtype=np.uint8)
        patch_rgba = np.dstack((patch_rgb, alpha))

        records.append(
            {
                "fi": int(fi),
                "patch": patch_rgba,
                "pw": c1 - c0,
                "ph": r1 - r0,
                "v0_px": (cols[0] - c0, rows[0] - r0),
                "v1_px": (cols[1] - c0, rows[1] - r0),
                "v2_px": (cols[2] - c0, rows[2] - r0),
            }
        )
    return records


def build_ortho(
    mesh: trimesh.Trimesh, terrain, epsg: int, reporter: Reporter | None = None
) -> Ortho:
    """NAIP over the terrain grid, roof offset found, roofs painted out of the ground."""
    reporter = reporter or get_reporter()
    ortho = fetch_naip(
        terrain.xs[0],
        terrain.ys[0],
        terrain.xs[-1],
        terrain.ys[-1],
        epsg,
        reporter=reporter,
    )
    footprints = roof_mask(mesh, ortho)
    dx, dy = estimate_roof_offset(ortho.image, footprints, ortho.res, reporter=reporter)
    ortho.roof_dx, ortho.roof_dy = dx, dy
    seen = footprints | roof_mask(mesh, ortho, dx, dy)
    ortho.ground = remove_roofs(ortho.image, seen, ortho.res)
    return ortho


def naip_stage(value, state: PipelineState):
    """Fetch and align NAIP when --naip is on; pass the building mesh through."""
    if not state.get_metadata("use_naip", False):
        return value
    reporter = get_reporter(state)
    terrain = state.get_metadata("terrain", None)
    if terrain is None:
        reporter.warning(
            "NAIP needs the terrain grid; continuing without aerial photos"
        )
        return value
    bbox = state.require_metadata("bbox")
    to_utm = _get_utm_transformer(
        (bbox.min_lon + bbox.max_lon) / 2, (bbox.min_lat + bbox.max_lat) / 2
    )
    try:
        ortho = build_ortho(value, terrain, to_utm.target_crs.to_epsg(), reporter)
        state.set_metadata("naip", ortho)
    except Exception as exc:  # noqa: BLE001 - plain ground and grey roofs still work
        reporter.warning(f"NAIP failed ({exc}); continuing without aerial photos")
    return value
