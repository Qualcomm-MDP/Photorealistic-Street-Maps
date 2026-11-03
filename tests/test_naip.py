import numpy as np
import pytest
import trimesh

from common import ProgressMonitor
from common.constants import BoundingBox
from common.pipeline_chain import PipelineState
from mesh_builder.extrude import extrude_buildings
from naip import Ortho, naip_stage
from naip import naip as naip_module
from naip.naip import estimate_roof_offset, remove_roofs, roof_faces, roof_mask
from pipeline import export_mesh
from terrain import Terrain
from texturing.tex_projection import UNTEXTURED_GREY, apply_textures

BBOX = BoundingBox(42.2830, 42.2850, -83.7485, -83.7455)
RES = 0.5
WHITE, BLUE = (255, 255, 255), (20, 40, 200)


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
        "tags": {"building": "yes", "height": "10"},
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in ring],
    }


@pytest.fixture
def mesh():
    return extrude_buildings({"elements": [_building()]}, BBOX)


def _ortho(mesh, roof_at=(0.0, 0.0), size_px=200) -> Ortho:
    """White aerial photo with the building's roof drawn blue, shifted by roof_at."""
    cx, cy = mesh.vertices[:, 0].mean(), mesh.vertices[:, 1].mean()
    half = size_px * RES / 2
    ortho = Ortho(
        image=np.full((size_px, size_px, 3), WHITE, dtype=np.uint8),
        west=cx - half,
        north=cy + half,
        res=RES,
    )
    ortho.image[roof_mask(mesh, ortho, *roof_at)] = BLUE
    return ortho


def test_roof_faces_are_the_top_of_the_building(mesh):
    roof = roof_faces(mesh)
    z_up = -mesh.triangles_center[:, 2]
    assert len(roof) == 2  # one rectangle = two triangles
    assert np.allclose(z_up[roof], z_up.max())


def test_drape_maps_image_corners_to_uv_corners(mesh):
    ortho = _ortho(mesh)
    corners = np.array(
        [
            [ortho.west, ortho.north, 0.0],
            [ortho.west + ortho.width_m, ortho.north - ortho.height_m, 0.0],
        ]
    )
    assert ortho.drape(corners).uv.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_roof_offset_is_recovered(mesh):
    ortho = _ortho(mesh, roof_at=(3.0, -2.0))
    dx, dy = estimate_roof_offset(ortho.image, roof_mask(mesh, ortho), RES)
    assert dx == pytest.approx(3.0, abs=RES)
    assert dy == pytest.approx(-2.0, abs=RES)


def test_roof_offset_at_the_search_limit_is_ignored(mesh):
    # 8 m: no outline edge lines up with a photo edge within +-5 m
    ortho = _ortho(mesh, roof_at=(8.0, 0.0))
    with pytest.warns(UserWarning, match="search limit"):
        offset = estimate_roof_offset(
            ortho.image, roof_mask(mesh, ortho), RES, search_m=5.0
        )
    assert offset == (0.0, 0.0)


def test_ground_has_the_roof_painted_out(mesh):
    ortho = _ortho(mesh)
    roof = roof_mask(mesh, ortho)
    ground = remove_roofs(ortho.image, roof, RES)
    assert np.all(ground[roof].min(axis=1) > 200)  # near white, not blue


def _texel(textured, face):
    """Texture colour at the centre of a face of a textured mesh."""
    uv = textured.visual.uv[textured.faces[face]].mean(axis=0)
    img = np.asarray(textured.visual.material.baseColorTexture)
    h, w = img.shape[:2]
    return tuple(int(c) for c in img[int((1 - uv[1]) * h), int(uv[0] * w), :3])


@pytest.mark.parametrize("roof_at", [(0.0, 0.0), (3.0, -2.0)])
def test_roofs_sample_the_aerial_photo_at_the_offset(mesh, roof_at):
    ortho = _ortho(mesh, roof_at=roof_at)
    ortho.roof_dx, ortho.roof_dy = roof_at

    textured = apply_textures(mesh, {}, BBOX, remove_obs=False, aerial=ortho)

    for fi in roof_faces(mesh):
        assert _texel(textured, fi) == BLUE
    walls = np.abs(mesh.face_normals[:, 2]) < 0.1
    assert _texel(textured, int(np.argmax(walls))) == (UNTEXTURED_GREY,) * 3


def _state(**metadata):
    return PipelineState(initial_input=None, current_value=None, metadata=metadata)


@pytest.fixture
def flat_terrain(mesh):
    lo, hi = mesh.bounds[0], mesh.bounds[1]
    xs = np.arange(lo[0] - 40, hi[0] + 40, 2.0)
    ys = np.arange(lo[1] - 40, hi[1] + 40, 2.0)
    return Terrain(xs=xs, ys=ys, z=np.zeros((len(ys), len(xs))), z_datum=250.0)


def _no_download(*args, **kwargs):
    raise AssertionError("should not download NAIP")


def test_naip_stage_does_nothing_when_disabled(monkeypatch, mesh, flat_terrain):
    monkeypatch.setattr(naip_module, "build_ortho", _no_download)
    state = _state(bbox=BBOX, terrain=flat_terrain)

    assert naip_stage(mesh, state) is mesh
    assert state.get_metadata("naip", None) is None


def test_naip_stage_needs_terrain(monkeypatch, mesh):
    monkeypatch.setattr(naip_module, "build_ortho", _no_download)
    state = _state(bbox=BBOX, use_naip=True)

    with pytest.warns(UserWarning, match="terrain"):
        assert naip_stage(mesh, state) is mesh
    assert state.get_metadata("naip", None) is None


def test_naip_stage_continues_without_photos_on_failure(
    monkeypatch, mesh, flat_terrain
):
    def fail(*args, **kwargs):
        raise ConnectionError("image server down")

    monkeypatch.setattr(naip_module, "build_ortho", fail)
    state = _state(bbox=BBOX, use_naip=True, terrain=flat_terrain)

    with pytest.warns(UserWarning, match="without aerial photos"):
        assert naip_stage(mesh, state) is mesh
    assert state.get_metadata("naip", None) is None


def test_export_drapes_the_photo_on_the_ground(tmp_path, mesh, flat_terrain):
    path = tmp_path / "model.glb"
    state = _state(
        bbox=BBOX,
        terrain=flat_terrain,
        naip=_ortho(mesh),
        output_path=str(path),
        progress_monitor=ProgressMonitor(verbose=False),
    )
    export_mesh(mesh, state)

    ground = trimesh.load(path).geometry["terrain"]
    assert ground.visual.uv is not None
    assert ground.visual.material.baseColorTexture is not None
