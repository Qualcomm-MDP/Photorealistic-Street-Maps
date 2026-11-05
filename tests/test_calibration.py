"""The calibrate stage (Mapillary's focal, distortion and gravity), the levelling fallbacks and the
gravity check, without COLMAP."""

import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from pose_alignment.align import level_camera
from pose_alignment.calibrate import (
    DEFAULT_FOCAL,
    calibrate_roi,
    choose_calibrations,
    focal_warnings,
    is_current,
    mapillary_gravity,
    rp_from_up,
    up_from_rotation,
)
from pose_alignment.geo_common import RoiPaths
from pose_alignment.gravity_check import spread
from pose_alignment.reconstruct import camera_setup

# A level camera looking north: columns are its right (east), down and forward (north) axes in
# ENU, i.e. world-from-camera. Mapillary's computed_rotation is world-to-camera.
LOOK_NORTH = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def mapillary_rotvec(world_from_camera):
    return Rotation.from_matrix(np.asarray(world_from_camera).T).as_rotvec().tolist()


def pitched(deg, heading_deg=0.0):
    """World-from-camera of a camera looking up by deg, turned clockwise by heading_deg."""
    up = Rotation.from_euler("x", deg, degrees=True).as_matrix()  # about its right axis
    turn = Rotation.from_euler("z", -heading_deg, degrees=True).as_matrix()
    return turn @ LOOK_NORTH @ up


def make_roi(tmp_path, sequences):
    """An ROI folder with sequences {folder: [frame dicts]} (and a tiny image per frame)."""
    p = RoiPaths(tmp_path / "mapillary_roi_1")
    p.images.mkdir(parents=True)
    p.sequences_json.write_text(json.dumps([{"folder": f} for f in sequences]))
    for folder, frames in sequences.items():
        (p.images / folder).mkdir()
        for k, fr in enumerate(frames):
            fr.setdefault("file", f"frame_{k + 1:04d}.jpg")
            Image.new("RGB", (64, 48)).save(p.images / folder / fr["file"])
        (p.images / folder / "cameras.json").write_text(json.dumps({"frames": frames}))
    return p


def frames(n=3, focal=0.8, make="Google", model="Pixel 3", rotation=True, pitch=0.0):
    return [
        {
            "make": make,
            "model": model,
            "width": 64,
            "height": 48,
            "camera_parameters": None if focal is None else [focal, -0.01, 0.002],
            "computed_rotation": mapillary_rotvec(pitched(pitch)) if rotation else None,
        }
        for _ in range(n)
    ]


# ----------------------------------------------------------------------------- gravity


def test_level_camera_has_up_along_minus_y():
    up = up_from_rotation(mapillary_rotvec(LOOK_NORTH))
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0], atol=1e-9)
    assert rp_from_up(up) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_pitch_is_positive_looking_up():
    roll, pitch = rp_from_up(up_from_rotation(mapillary_rotvec(pitched(10.0))))
    assert pitch == pytest.approx(10.0, abs=1e-6)
    assert roll == pytest.approx(0.0, abs=1e-6)


def test_mapillary_gravity_interpolates_frames_without_a_rotation():
    files = [f"frame_{i:04d}.jpg" for i in range(1, 6)]
    meta = {f: {} for f in files}
    meta[files[0]]["computed_rotation"] = mapillary_rotvec(pitched(0.0))
    meta[files[2]]["computed_rotation"] = mapillary_rotvec(pitched(4.0))
    out, n_own = mapillary_gravity(meta, files)

    assert n_own == 2
    assert [out[f]["estimated"] for f in files] == [True, False, True, False, False]
    assert out[files[1]]["pitch_deg"] == pytest.approx(2.0, abs=0.05)  # between 0 and 4
    assert out[files[4]]["pitch_deg"] == pytest.approx(
        4.0, abs=1e-3
    )  # nearest at the end


def test_mapillary_gravity_is_none_without_rotations():
    assert mapillary_gravity({"a.jpg": {}}, ["a.jpg"]) is None


# ----------------------------------------------------------------------------- focal check

PIXEL = ("Google", "Pixel 3", 64, 48)


def test_a_focal_far_from_its_cameras_median_is_replaced():
    mly = {
        "a": [0.73, -0.01, 0.0],
        "b": [0.74, -0.02, 0.0],
        "c": [0.75, -0.03, 0.0],
        "d": [0.976, 0.0, -0.003],  # 32% above the others' median
    }
    out = choose_calibrations(mly, {f: PIXEL for f in mly})

    assert [out[f][0] for f in "abc"] == ["mapillary"] * 3
    source, calib, why = out["d"]
    assert source == "mapillary_median"
    assert calib == [0.74, -0.02, 0.0]  # the sequence closest to the median, as a whole
    assert "0.976" in why and "0.740" in why


def test_focal_check_needs_two_others_of_the_same_camera():
    two = {"a": [0.73, 0.0, 0.0], "b": [0.98, 0.0, 0.0]}
    assert {
        s for s, _, _ in choose_calibrations(two, {f: PIXEL for f in two}).values()
    } == {"mapillary"}
    mixed = {"a": [0.73, 0.0, 0.0], "b": [0.74, 0.0, 0.0], "c": [0.4, 0.0, 0.0]}
    cameras = {"a": PIXEL, "b": PIXEL, "c": ("Apple", "iPhone", 64, 48)}
    assert choose_calibrations(mixed, cameras)["c"][0] == "mapillary"


def test_two_disagreeing_sequences_are_kept_with_a_warning():
    two = {"a": [1.129, 0.0, 0.0], "b": [1.679, 0.0, 0.0]}
    cameras = {f: ("Suzuki", "DL1000", 64, 48) for f in two}
    assert {s for s, _, _ in choose_calibrations(two, cameras).values()} == {
        "mapillary"
    }
    (warning,) = focal_warnings(two, cameras)
    assert "1.129" in warning and "1.679" in warning and "Suzuki DL1000" in warning
    close = {"a": [0.73, 0.0, 0.0], "b": [0.75, 0.0, 0.0]}
    assert focal_warnings(close, {f: PIXEL for f in close}) == []


def test_no_mapillary_calibration_starts_from_the_default_focal():
    assert choose_calibrations({"a": None}, {"a": PIXEL})["a"] == (
        "default",
        [DEFAULT_FOCAL, 0.0, 0.0],
        None,
    )


# ----------------------------------------------------------------------------- the stage


def test_calibrate_roi_writes_mapillary_calibrations(tmp_path):
    p = make_roi(
        tmp_path,
        {
            "line1_cam0_a": frames(focal=0.8, pitch=3.0),
            "line1_cam1_b": frames(focal=None),
            "line1_cam2_c": frames(rotation=False),
        },
    )
    calibrate_roi(p.root)

    a = json.loads(p.calibration_json("line1_cam0_a").read_text())
    assert a["source"] == "mapillary" and a["gravity_source"] == "mapillary"
    assert a["focal_px"] == pytest.approx(0.8 * 64)
    assert a["colmap_params"] == pytest.approx([0.8 * 64, 32, 24, -0.01, 0.002])
    assert a["frames"]["frame_0001.jpg"]["pitch_deg"] == pytest.approx(3.0, abs=1e-3)
    assert is_current(a)

    b = json.loads(p.calibration_json("line1_cam1_b").read_text())
    assert b["source"] == "default" and b["focal_px"] == pytest.approx(1.2 * 64)

    c = json.loads(p.calibration_json("line1_cam2_c").read_text())
    assert c["gravity_source"] is None and c["frames"] == {}  # no rotations: no gravity
    assert is_current(c)


def test_geocalib_calibrations_are_recalculated(tmp_path, capsys):
    p = make_roi(tmp_path, {"line1_cam0_a": frames()})
    out = p.calibration_json("line1_cam0_a")
    out.parent.mkdir(parents=True)
    old = {"source": "geocalib", "focal_norm": 0.5, "k1": 0.0, "frames": {}}
    out.write_text(json.dumps(old))
    assert not is_current(old)
    assert not is_current({"source": "mapillary", "frames": {}})  # GeoCalib's gravity

    calibrate_roi(p.root)
    assert "GeoCalib version" in capsys.readouterr().out
    assert json.loads(out.read_text())["source"] == "mapillary"

    calibrate_roi(p.root)  # current now: reused
    assert "calibration exists" in capsys.readouterr().out


def test_camera_setup_takes_the_calibration(tmp_path):
    p = make_roi(tmp_path, {"line1_cam0_a": frames(focal=0.8)})
    calibrate_roi(p.root)

    model, params, info = camera_setup(p, "line1_cam0_a")
    assert model == "RADIAL"
    assert params == pytest.approx([0.8 * 64, 32, 24, -0.01, 0.002])
    assert info["source"] == "mapillary"
    _, params, info = camera_setup(p, "line1_cam0_a", focal=0.7)
    assert (
        params[0] == pytest.approx(0.7 * 64) and "focal set to 0.700" in info["source"]
    )


# ----------------------------------------------------------------------------- levelling


def street(n=20):
    """Cameras along a street heading east (model frame = world, z up), all level."""
    R = np.array([pitched(0.0, heading_deg=90.0) for _ in range(n)])
    C = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.full(n, 1.6)])
    return [f"frame_{i:04d}.jpg" for i in range(n)], C, R


def test_level_camera_uses_gravity_then_the_road_then_the_cameras():
    names, C, R = street()
    rng = np.random.default_rng(1)
    road_pts = np.column_stack(
        [rng.uniform(0, 19, 200), rng.uniform(-4, 4, 200), rng.normal(0, 0.02, 200)]
    )
    gravity = {
        "frames": {
            n: {"up_cam": up_from_rotation(mapillary_rotvec(r)).tolist()}
            for n, r in zip(names, R)
        }
    }

    _, basis, _, _, src = level_camera(
        names, C, R, gravity, road_pts, np.ones(200, bool)
    )
    assert src == "Mapillary gravity"
    np.testing.assert_allclose(basis[2], [0, 0, 1], atol=1e-6)

    _, basis, spread_deg, _, src = level_camera(
        names, C, R, {"frames": {}}, road_pts, np.ones(200, bool)
    )
    assert src.startswith("road plane") and spread_deg is None
    np.testing.assert_allclose(
        basis[2], [0, 0, 1], atol=0.01
    )  # up, pointing at the cameras

    _, basis, _, _, src = level_camera(
        names, C, R, {"frames": {}}, road_pts, np.zeros(200, bool)
    )
    assert src.startswith("cameras' down axes")
    np.testing.assert_allclose(basis[2], [0, 0, 1], atol=1e-6)


def test_gravity_check_spread_reflects_per_frame_noise():
    rng = np.random.default_rng(0)
    names = [f"frame_{i:04d}.jpg" for i in range(40)]
    R = np.array([pitched(0.0, heading_deg=h) for h in np.linspace(0, 30, len(names))])
    C = np.column_stack(
        [np.arange(len(names)), np.zeros(len(names)), np.zeros(len(names))]
    )

    def gc(noise_deg):
        out = {}
        for n, Rwc in zip(names, R):
            tilt = Rotation.from_rotvec(np.radians(noise_deg) * rng.normal(size=3))
            out[n] = {
                "up_cam": up_from_rotation(
                    mapillary_rotvec(tilt.as_matrix() @ Rwc)
                ).tolist()
            }
        return {"frames": out}

    exact, _, used = spread(names, C, R, gc(0.0))
    noisy, _, _ = spread(names, C, R, gc(1.0))
    assert used == len(names)
    assert exact == pytest.approx(0.0, abs=1e-4)
    assert 0.5 < noisy < 3.0
