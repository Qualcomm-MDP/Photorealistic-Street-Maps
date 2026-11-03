import numpy as np
import pytest
import trimesh

from common import ProgressMonitor
from common.MeshUtils import _get_utm_transformer
from common.constants import BoundingBox
from common.pipeline_chain import PipelineState
from mesh_builder.extrude import EMBED_M, extrude_buildings
from pipeline import export_mesh
from terrain import Terrain, terrain_stage
from terrain import terrain as terrain_module
from terrain.terrain import decode_terrarium
from texturing.tex_projection import CAMERA_HEIGHT_M, camera_position

BBOX = BoundingBox(42.2830, 42.2850, -83.7485, -83.7455)
SLOPE = 0.1  # ground rises 10 cm per metre east
BUILDING_HEIGHT = 10.0


def _building():
    lon0, lat0, dlon, dlat = -83.7472, 42.2838, 0.0002, 0.0001
    ring = [
        (lon0, lat0),
        (lon0 + dlon, lat0),
        (lon0 + dlon, lat0 + dlat),
        (lon0, lat0 + dlat),
        (lon0, lat0),
    ]
    return {
        "type": "way",
        "id": 1,
        "tags": {"building": "yes", "height": str(BUILDING_HEIGHT)},
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in ring],
    }


def _utm_corners(element):
    to_utm = _get_utm_transformer(
        (BBOX.min_lon + BBOX.max_lon) / 2, (BBOX.min_lat + BBOX.max_lat) / 2
    )
    return np.array([to_utm.transform(p["lon"], p["lat"]) for p in element["geometry"]])


@pytest.fixture
def sloped_terrain():
    corners = _utm_corners(_building())
    xs = np.arange(corners[:, 0].min() - 60, corners[:, 0].max() + 60, 2.0)
    ys = np.arange(corners[:, 1].min() - 60, corners[:, 1].max() + 60, 2.0)
    XX, _ = np.meshgrid(xs, ys)
    return Terrain(xs=xs, ys=ys, z=SLOPE * (XX - xs[0]), z_datum=250.0)


def test_decode_terrarium_matches_the_published_formula():
    # 250.5 m = 32768 + 250.5 = 33018.5 -> R=128, G=250, B=128
    rgb = np.array([[[128, 250, 128], [128, 0, 0]]], dtype=np.uint8)
    assert decode_terrarium(rgb).tolist() == [[250.5, 0.0]]


def test_terrain_ground_interpolates_between_grid_points(sloped_terrain):
    x = sloped_terrain.xs[0] + 7.0  # between grid points
    y = sloped_terrain.ys[3]
    assert float(sloped_terrain.ground(x, y)) == pytest.approx(SLOPE * 7.0)


def test_flat_ground_is_unchanged_without_terrain():
    mesh = extrude_buildings({"elements": [_building()]}, BBOX)
    z_up = -mesh.vertices[:, 2]  # build_mesh output is z-down
    assert z_up.min() == pytest.approx(0.0)
    assert z_up.max() == pytest.approx(BUILDING_HEIGHT)


def test_building_on_a_slope_spans_lowest_to_highest_ground(sloped_terrain):
    corners = _utm_corners(_building())
    ground = SLOPE * (corners[:, 0] - sloped_terrain.xs[0])

    mesh = extrude_buildings({"elements": [_building()]}, BBOX, terrain=sloped_terrain)
    z_up = -mesh.vertices[:, 2]

    assert z_up.min() == pytest.approx(ground.min() - EMBED_M, abs=1e-6)
    assert z_up.max() == pytest.approx(ground.max() + BUILDING_HEIGHT, abs=1e-6)


def test_camera_sits_above_the_ground_at_its_position(sloped_terrain):
    x, y = sloped_terrain.xs[10], sloped_terrain.ys[10]
    flat = camera_position(x, y)
    on_hill = camera_position(x, y, sloped_terrain)

    assert flat[2] == pytest.approx(CAMERA_HEIGHT_M)
    assert on_hill[2] == pytest.approx(SLOPE * 20.0 + CAMERA_HEIGHT_M)


def _state(**metadata):
    return PipelineState(initial_input=None, current_value=None, metadata=metadata)


def test_terrain_stage_does_nothing_when_disabled(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("should not download terrain")

    monkeypatch.setattr(terrain_module, "build_terrain", fail)
    state = _state(bbox=BBOX)

    assert terrain_stage("fetched", state) == "fetched"
    assert state.get_metadata("terrain", None) is None


def test_terrain_stage_falls_back_to_flat_ground_on_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("tile server down")

    monkeypatch.setattr(terrain_module, "build_terrain", fail)
    state = _state(bbox=BBOX, use_terrain=True)

    with pytest.warns(UserWarning, match="flat ground"):
        assert terrain_stage("fetched", state) == "fetched"
    assert state.get_metadata("terrain", None) is None


def _export(tmp_path, terrain):
    mesh = extrude_buildings({"elements": [_building()]}, BBOX, terrain=terrain)
    path = tmp_path / "model.glb"
    state = _state(
        bbox=BBOX,
        terrain=terrain,
        output_path=str(path),
        progress_monitor=ProgressMonitor(verbose=False),
    )
    export_mesh(mesh, state)
    return trimesh.load(path)


def test_export_without_terrain_is_a_single_mesh(tmp_path):
    scene = _export(tmp_path, terrain=None)
    assert len(scene.geometry) == 1


def test_export_with_terrain_adds_the_ground_in_the_same_frame(
    tmp_path, sloped_terrain
):
    scene = _export(tmp_path, sloped_terrain)
    assert set(scene.geometry) == {"buildings", "terrain"}

    # glTF is Y-up: the building's base must sit on the ground surface, not
    # float above it or sink below the lowest ground.
    buildings = scene.geometry["buildings"].bounds
    ground = scene.geometry["terrain"].bounds
    assert ground[0, 1] - EMBED_M - 1e-6 <= buildings[0, 1] <= ground[1, 1]
