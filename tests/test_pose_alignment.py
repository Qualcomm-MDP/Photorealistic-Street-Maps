"""Pose alignment logic that runs without COLMAP, GeoCalib or LightGlue."""

import json
import sqlite3

import numpy as np

from pose_alignment.align_poses import _apply_alignment
from pose_alignment.geo_common import RoiPaths
from pose_alignment.mapillary_layer import crop_axes
from pose_alignment.reconstruct import weak_ends

# A camera looking north: columns are its right (east), down and forward (north) axes in ENU.
LOOK_NORTH = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
LINE = [[-83.7400, 42.2800], [-83.7390, 42.2800]]


def test_crop_axes_turns_the_view_right():
    R = crop_axes(90.0)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        R[:, 2], [1.0, 0.0, 0.0], atol=1e-12
    )  # forward = the 360 camera's right
    np.testing.assert_allclose(crop_axes(-90.0)[:, 2], [-1.0, 0.0, 0.0], atol=1e-12)


def _matches_db(path, n, links):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
    con.executemany(
        "INSERT INTO images VALUES (?, ?)", [(i + 1, f"f{i:02d}.jpg") for i in range(n)]
    )
    for (a, b), rows in links.items():
        i, j = sorted((a + 1, b + 1))
        con.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?)", (i * 2147483647 + j, rows)
        )
    con.commit()
    con.close()


def test_weak_ends_drops_a_tail_held_by_few_matches(tmp_path):
    n = 30
    links = {(k, k + 1): 300 for k in range(n - 1)} | {
        (k, k + 2): 200 for k in range(n - 2)
    }
    links |= {
        (26, 27): 20,
        (25, 27): 10,
        (26, 28): 10,
    }  # frames 27-29 hang on by a few matches
    _matches_db(tmp_path / "db.db", n, links)
    groups = [[f"f{i:02d}.jpg"] for i in range(n)]
    keep, dropped, info = weak_ends(tmp_path / "db.db", groups)
    assert dropped == ["f27.jpg", "f28.jpg", "f29.jpg"]
    assert keep == [f"f{i:02d}.jpg" for i in range(27)]
    assert info["back_frames_dropped"] == 3 and info["front_frames_dropped"] == 0


def test_weak_ends_keeps_an_evenly_matched_leg(tmp_path):
    n = 30
    _matches_db(tmp_path / "db.db", n, {(k, k + 1): 300 for k in range(n - 1)})
    keep, dropped, _ = weak_ends(
        tmp_path / "db.db", [[f"f{i:02d}.jpg"] for i in range(n)]
    )
    assert len(keep) == n and dropped == []


def _camera(img_id, file, lon, **extra):
    return {
        "file": file,
        "id": img_id,
        "leg": "leg01",
        "lon": lon,
        "lat": 42.2800,
        "height_m": 2.1,
        "rotation_cam_to_enu": LOOK_NORTH,
        "intrinsics": {
            "model": "SIMPLE_RADIAL",
            "width": 1000,
            "height": 750,
            "fx": 800.0,
        },
        "leg_fit_m": 0.8,
        "reliable": True,
        **extra,
    }


def _roi(tmp_path, alignment):
    paths = RoiPaths(tmp_path / "mapillary_roi_1")
    paths.alignment.mkdir(parents=True)
    paths.images.mkdir(parents=True)
    paths.roi_json.write_text(json.dumps({"lines": [LINE], "corridor_m": 15}))
    paths.sequences_json.write_text(json.dumps([{"folder": f} for f in alignment]))
    for folder, cams in alignment.items():
        (paths.alignment / f"{folder}.json").write_text(
            json.dumps({"folder": folder, "cameras": cams})
        )
    (paths.alignment / "summary.json").write_text("{}")  # not a sequence: skipped
    return paths


def test_apply_alignment_overrides_and_adds_360_crops(tmp_path):
    crop = {
        "intrinsics": {
            "model": "SIMPLE_RADIAL",
            "width": 960,
            "height": 960,
            "fx": 480.0,
        }
    }
    paths = _roi(
        tmp_path,
        {
            "line1_cam0_aaaaaaaa": [_camera("111", "frame_0001.jpg", -83.7398)],
            # a 360 frame's right and left crops share the Mapillary id
            "line2_cam0_pppppppp": [
                _camera("222", "frame_0001.jpg", -83.7395, pano_yaw_deg=90.0, **crop)
            ],
            "line2_cam1_pppppppp": [
                _camera("222", "frame_0001.jpg", -83.7395, pano_yaw_deg=-90.0, **crop)
            ],
        },
    )
    mapillary = {
        "111": {
            "latitude": 42.2801,
            "longitude": -83.7398,
            "original_width": 4000,
            "original_height": 3000,
        },
        "222": {
            "latitude": 42.2801,
            "longitude": -83.7395,
            "original_width": 5760,
            "original_height": 2880,
        },
    }
    summary = _apply_alignment(paths, mapillary)

    assert summary["overridden"] == 1 and summary["added"] == 2
    assert mapillary["111"]["pose_source"] == "colmap"
    assert (
        mapillary["111"]["focal_length_pixels"] == 800.0 * 4000 / 1000
    )  # scaled to the original size
    assert (
        "pose_source" not in mapillary["222"]
    )  # the 360 image itself keeps Mapillary's pose
    for side in ("line2_cam0_pppppppp", "line2_cam1_pppppppp"):
        added = mapillary[f"222_{side}"]
        assert added["image_url"].endswith(f"{side}/frame_0001.jpg")
        assert added["original_width"] == 960 and added["focal_length_pixels"] == 480.0
    overrides = json.loads((paths.root / "pose_overrides.json").read_text())
    assert overrides["images"]["111"]["leg_fit_to_walls_m"] == 0.8


def test_apply_alignment_skips_unreliable_and_off_street_legs(tmp_path):
    far = _camera("333", "frame_0001.jpg", -83.7398)
    far["lat"] = 42.2810  # about 110 m north of the street
    paths = _roi(
        tmp_path,
        {
            "line1_cam0_aaaaaaaa": [
                _camera("111", "frame_0001.jpg", -83.7398, reliable=False)
            ],
            "line1_cam1_bbbbbbbb": [far],
        },
    )
    mapillary = {
        i: {
            "latitude": 42.28,
            "longitude": -83.7398,
            "original_width": 1000,
            "original_height": 750,
        }
        for i in ("111", "333")
    }
    summary = _apply_alignment(paths, mapillary)
    assert summary["overridden"] == 0 and summary["rejected_legs_off_street_m"]
    assert all("pose_source" not in m for m in mapillary.values())
