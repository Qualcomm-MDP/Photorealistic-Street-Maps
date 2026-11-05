"""Pose alignment logic that runs without COLMAP, GeoCalib or LightGlue."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from pose_alignment.align_poses import _apply_alignment
from pose_alignment.geo_common import RoiPaths
from data_ingest.pano import crop_axes
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
        # the fetch stage's own views of the 360 photo (data_ingest.pano), Mapillary's pose
        **{
            f"222_{side}": {
                "latitude": 42.2801,
                "longitude": -83.7395,
                "original_width": 1440,
                "original_height": 1440,
                "pano_crop": {"id": "222", "side": side},
            }
            for side in ("right", "left")
        },
    }
    summary = _apply_alignment(paths, mapillary)

    assert summary["overridden"] == 1 and summary["added"] == 2
    assert mapillary["111"]["pose_source"] == "colmap"
    assert (
        mapillary["111"]["focal_length_pixels"] == 800.0 * 4000 / 1000
    )  # scaled to the original size
    # the aligned views replace the fetch stage's unaligned ones
    assert "222_right" not in mapillary and "222_left" not in mapillary
    for side in ("line2_cam0_pppppppp", "line2_cam1_pppppppp"):
        added = mapillary[f"222_{side}"]
        assert Path(added["image_url"]).parts[-2:] == (side, "frame_0001.jpg")
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


def test_pano_views_turn_a_360_photo_into_right_and_left_views():
    from data_ingest.pano import crop_rotvec, pano_views

    # a 360 sequence driving east; the image centre looks along travel
    records = [
        {
            "id": str(k),
            "sequence": "s360",
            "camera_type": "spherical",
            "captured_at": k * 1000,
            "computed_compass_angle": 90.0,
            "width": 5760,
            "geometry": {"coordinates": [-83.74 + k * 1e-4, 42.28]},
        }
        for k in range(3)
    ] + [{"id": "9", "camera_type": "perspective", "geometry": {"coordinates": [0, 0]}}]
    rot = [0.3, -1.1, 0.4]
    metadata = {
        r["id"]: {
            "rotvec": rot,
            "original_width": r.get("width"),
            "camera_type": r["camera_type"],
        }
        for r in records
    }
    out = pano_views(metadata, records)
    assert (
        "1" not in out and "9" in out
    )  # the 360 photo is replaced, a perspective one kept
    right, left = out["1_right"], out["1_left"]
    assert right["pano_crop"]["yaw_deg"] == pytest.approx(90.0, abs=1)
    assert left["pano_crop"]["yaw_deg"] == pytest.approx(-90.0, abs=1)
    assert (
        right["original_width"] == right["original_height"] == 1440
    )  # 5760 * 90 / 360
    assert right["focal_length_pixels"] == pytest.approx(720.0)  # 90 deg across
    np.testing.assert_allclose(
        right["rotvec"], crop_rotvec(rot, right["pano_crop"]["yaw_deg"])
    )


def test_crop_view_looks_where_its_yaw_says():
    from data_ingest.pano import crop_view

    # an equirectangular image whose blue channel is a quarter of its column
    pano = np.zeros((400, 800, 3), np.uint8)
    pano[..., 0] = (np.arange(800) // 4)[None, :]
    centre = lambda yaw: int(crop_view(pano, yaw, 90.0, 100)[50, 50, 0])  # noqa: E731
    assert abs(centre(90.0) - 150) <= 1  # +90 deg: column 600, three quarters across
    assert abs(centre(-90.0) - 50) <= 1  # -90 deg: column 200, one quarter across


def _rig_entry(fit, points=1500, reliable=None, backward=()):
    return (
        {
            "aligned": True,
            "reliable": fit <= 2.0 if reliable is None else reliable,
            "points_to_footprint_median_m": {"aligned": fit},
            "building_points": points,
            "cameras_out_of_order": list(backward),
            "cameras_inside_buildings": 0,
            "registered": 40,
        },
        [{"file": "frame_0001.jpg", "reliable": fit <= 2.0}],
    )


def test_a_rig_is_reliable_when_its_best_camera_fits():
    from pose_alignment.align import rig_reliability

    leg = {
        "id": "leg04",
        "cameras": [
            {"folder": "right", "rig": "right"},
            {"folder": "left", "rig": "right"},
        ],
    }
    out = {"right": _rig_entry(3.0), "left": _rig_entry(1.9)}
    rig_reliability(leg, out, 2.0)
    assert out["right"][0]["reliable"] and out["right"][1][0]["reliable"]
    assert "reliable_via_rig" in out["right"][0]

    # both sides far off: the shared pose is wrong
    out = {"right": _rig_entry(4.5), "left": _rig_entry(1.9)}
    rig_reliability(leg, out, 2.0)
    assert not out["right"][0]["reliable"]
    # a folded reconstruction stays unreliable however well it fits
    out = {
        "right": _rig_entry(3.0, backward=["frame_0002.jpg"]),
        "left": _rig_entry(1.9),
    }
    rig_reliability(leg, out, 2.0)
    assert not out["right"][0]["reliable"]
