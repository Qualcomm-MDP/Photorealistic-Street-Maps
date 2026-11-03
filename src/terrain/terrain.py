from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import requests
import trimesh
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from trimesh.visual.material import PBRMaterial

from common import PipelineState
from common.MeshUtils import _get_utm_transformer
from common.constants import BoundingBox
from common.reporter import Reporter, get_reporter

TERRARIUM_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
TERRARIUM_ZOOM = 15  # max zoom, ~3.5 m/px at Ann Arbor's latitude
GRID_SPACING_M = 2.0
# Overpass returns whole buildings that cross the bbox edge, so the grid has
# to reach past the bbox or their ground heights would be extrapolated.
MARGIN_M = 100.0
TERRAIN_COLOR = [150, 155, 130, 255]


@dataclass
class Terrain:
    """Ground heights on a regular UTM grid.

    xs, ys are absolute UTM coordinates (same frame as the building meshes);
    z[row, col] is the height at (xs[col], ys[row]) in metres above z_datum,
    the lowest point of the grid.
    """

    xs: np.ndarray
    ys: np.ndarray
    z: np.ndarray
    z_datum: float

    def __post_init__(self) -> None:
        self._interp = RegularGridInterpolator(
            (self.ys, self.xs), self.z, bounds_error=False, fill_value=None
        )

    @property
    def spacing(self) -> float:
        return float(self.xs[1] - self.xs[0])

    def ground(self, x, y) -> np.ndarray:
        """Ground height (m above z_datum) at UTM points x, y."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        return self._interp(np.column_stack([y.ravel(), x.ravel()])).reshape(x.shape)

    def to_mesh(self) -> trimesh.Trimesh:
        """Z-up grid mesh of the ground in absolute UTM x, y."""
        XX, YY = np.meshgrid(self.xs, self.ys)
        ny, nx = XX.shape
        verts = np.column_stack([XX.ravel(), YY.ravel(), self.z.ravel()])
        idx = np.arange(ny * nx).reshape(ny, nx)
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        faces = np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        # Without an explicit material glTF viewers default to metallic=1,
        # which renders the ground nearly black.
        mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(
                name="terrain",
                baseColorFactor=TERRAIN_COLOR,
                metallicFactor=0.0,
                roughnessFactor=1.0,
            )
        )
        return mesh


def _lonlat_to_global_px(lon, lat, zoom: int, tile: int = 256):
    n = 2**zoom
    x = (np.asarray(lon) + 180.0) / 360.0 * n * tile
    lat_r = np.radians(np.asarray(lat))
    y = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n * tile
    return x, y


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """Terrarium PNG pixels (H, W, 3) -> elevation in metres."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


def terrarium_sampler(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    reporter: Reporter | None = None,
):
    """Download the Terrarium tiles covering the area; return sample(lon, lat) -> m."""
    reporter = reporter or get_reporter()
    gx0, gy0 = _lonlat_to_global_px(west, north, zoom)
    gx1, gy1 = _lonlat_to_global_px(east, south, zoom)
    # one-pixel margin so bilinear sampling at the edge stays inside the mosaic
    tx0, ty0 = int((gx0 - 1) // 256), int((gy0 - 1) // 256)
    tx1, ty1 = int((gx1 + 1) // 256), int((gy1 + 1) // 256)

    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), dtype=np.float32)
    session = requests.Session()
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            r = session.get(TERRARIUM_URL.format(z=zoom, x=tx, y=ty), timeout=30)
            r.raise_for_status()
            rgb = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
            r0, c0 = (ty - ty0) * 256, (tx - tx0) * 256
            mosaic[r0 : r0 + 256, c0 : c0 + 256] = decode_terrarium(rgb)
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    reporter.info(f"  Downloaded {n_tiles} Terrarium elevation tile(s) at zoom {zoom}")

    def sample(lon, lat):
        gx, gy = _lonlat_to_global_px(lon, lat, zoom)
        cols = gx - tx0 * 256 - 0.5  # pixel centres sit at +0.5
        rows = gy - ty0 * 256 - 0.5
        return map_coordinates(mosaic, [rows, cols], order=1, mode="nearest")

    return sample


def build_terrain(
    bbox: BoundingBox,
    spacing: float = GRID_SPACING_M,
    margin: float = MARGIN_M,
    zoom: int = TERRARIUM_ZOOM,
    reporter: Reporter | None = None,
) -> Terrain:
    reporter = reporter or get_reporter()
    center_lon = (bbox.min_lon + bbox.max_lon) / 2
    center_lat = (bbox.min_lat + bbox.max_lat) / 2
    to_utm = _get_utm_transformer(center_lon, center_lat)
    to_ll = Transformer.from_crs(to_utm.target_crs, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(
        [bbox.min_lon, bbox.max_lon, bbox.max_lon, bbox.min_lon],
        [bbox.min_lat, bbox.min_lat, bbox.max_lat, bbox.max_lat],
    )
    xs = np.arange(min(cx) - margin, max(cx) + margin + 1e-6, spacing)
    ys = np.arange(min(cy) - margin, max(cy) + margin + 1e-6, spacing)

    lons, lats = to_ll.transform(
        [xs[0], xs[-1], xs[-1], xs[0]], [ys[0], ys[0], ys[-1], ys[-1]]
    )
    sample = terrarium_sampler(
        min(lons), min(lats), max(lons), max(lats), zoom, reporter
    )

    XX, YY = np.meshgrid(xs, ys)
    lon, lat = to_ll.transform(XX.ravel(), YY.ravel())
    Z = np.asarray(sample(lon, lat), dtype=np.float64).reshape(XX.shape)
    z_datum = float(Z.min())
    reporter.info(
        f"  Terrain grid {len(xs)}x{len(ys)} at {spacing:g} m, elevation "
        f"{Z.min():.1f}-{Z.max():.1f} m (relief {Z.max() - Z.min():.1f} m)"
    )
    return Terrain(xs=xs, ys=ys, z=Z - z_datum, z_datum=z_datum)


def terrain_stage(value, state: PipelineState):
    """Build the ground model when --terrain is on; pass the fetch output through."""
    if not state.get_metadata("use_terrain", False):
        return value
    reporter = get_reporter(state)
    try:
        terrain = build_terrain(state.require_metadata("bbox"), reporter=reporter)
        state.set_metadata("terrain", terrain)
    except Exception as exc:  # noqa: BLE001 - flat ground is the old behaviour
        reporter.warning(f"Terrain failed ({exc}); continuing with flat ground")
    return value
