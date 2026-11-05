"""Blended wall textures and the viewer overlays, on synthetic scenes."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.overlays import build_overlays
from common.reporter import Reporter
from texturing.blend import bake_walls

# A camera looking north: rows are its right (east), down and forward (north) axes.
LOOK_NORTH = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class _Quiet(Reporter):
    def info(self, message):
        pass

    def progress(self, items, desc, total=None):
        return iter(items)


def _wall():
    """A 4 m x 3 m wall 10 m north of the origin, facing south, as two triangles."""
    verts = np.array(
        [[0, 10, 0], [4, 10, 0], [4, 10, 3], [0, 10, 3]], float
    )  # projection space, z up
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    return verts, faces, normals


def test_both_triangles_of_a_wall_bake_the_same_photo(tmp_path):
    photo = np.zeros((300, 400, 3), np.uint8)
    photo[:, :200] = (0, 0, 255)  # left half red (BGR)
    photo[:, 200:] = (255, 0, 0)  # right half blue
    cv2.imwrite(str(tmp_path / "p.png"), photo)
    K = np.array([[200.0, 0, 200], [0, 200.0, 150], [0, 0, 1]])
    cam = {
        "img_path": tmp_path / "p.png",
        "w": 400,
        "h": 300,
        "K": K,
        "R": LOOK_NORTH,
        "Cw": np.array([2.0, 0.0, 1.5]),
        "mask": None,
    }
    verts, faces, normals = _wall()
    patches, stats = bake_walls(verts, faces, normals, [cam], set(), _Quiet())
    assert stats["textured_walls"] == 2 and stats["coverage"] > 0.99
    for p in patches:
        rgb = p["patch"]
        # world-aligned grid: the wall's west half (x < 2 m, left in the photo) is red
        assert (rgb[:, : p["pw"] // 2 - 2, 0] > 200).all()
        assert (rgb[:, p["pw"] // 2 + 2 :, 2] > 200).all()
    # the shared corner (vertex 0, x = 0, z = 0) maps to the same patch pixel in both
    assert patches[0]["v0_px"] == pytest.approx(patches[1]["v0_px"])


def test_obstructed_pixels_take_another_photo(tmp_path):
    K = np.array([[200.0, 0, 200], [0, 200.0, 150], [0, 0, 1]])
    cams = []
    for k, (colour, x) in enumerate((((0, 0, 255), 1.5), ((0, 255, 0), 2.5))):
        cv2.imwrite(
            str(tmp_path / f"{k}.png"), np.full((300, 400, 3), colour, np.uint8)
        )
        cams.append(
            {
                "img_path": tmp_path / f"{k}.png",
                "w": 400,
                "h": 300,
                "K": K,
                "R": LOOK_NORTH,
                "Cw": np.array([x, 0.0, 1.5]),
                # the first photo's view is blocked everywhere (a bus in front)
                "mask": np.full((38, 50), 255 if k == 0 else 0, np.uint8),
            }
        )
    verts, faces, normals = _wall()
    patches, _ = bake_walls(verts, faces, normals, cams, set(), _Quiet())
    for p in patches:
        assert (p["patch"][..., 1] > 200).all()  # green: the unobstructed photo


class _Transformer:
    def transform(self, lon, lat):
        return np.asarray(lon) * 1000.0, np.asarray(lat) * 1000.0


def test_overlays_are_in_the_models_frame():
    poses = {
        "1": {
            "latitude": 0.002,
            "longitude": 0.001,
            "rotvec": Rotation.from_matrix(LOOK_NORTH).as_rotvec().tolist(),
            "camera_height_m": 2.0,
            "focal_length_pixels": 500.0,
            "original_width": 1000,
            "original_height": 750,
            "pose_source": "colmap",
        }
    }
    points = {
        "lon": np.array([0.003]),
        "lat": np.array([0.004]),
        "height_m": np.array([5.0]),
        "rgb": np.array([[10, 20, 30]]),
    }
    out = build_overlays(poses, points, _Transformer(), (0.0, 0.0), (0.0, 0.0, 0.0))
    (cam,) = out["cameras"]
    # glTF y up: (x, y, z) -> (x, z, -y)
    assert cam["p"] == pytest.approx([1.0, 2.0, -2.0])
    assert cam["f"] == pytest.approx([0.0, 0.0, -1.0])  # north is -z
    assert cam["u"] == pytest.approx([0.0, 1.0, 0.0])
    assert cam["source"] == "colmap" and cam["fov"] == pytest.approx(73.7, abs=0.1)
    assert out["points"]["xyz"] == pytest.approx([3.0, 5.0, -4.0])
    assert out["points"]["rgb"] == [10, 20, 30]
