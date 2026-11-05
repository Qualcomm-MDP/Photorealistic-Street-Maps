"""The fetch stage's records for the pose alignment, and its downloads through the image
store, with Mapillary faked."""

import json

import cv2
import numpy as np
import pytest

from common.constants import BoundingBox
from data_ingest import image_store, ingest
from data_ingest.image_store import ImageStore
from data_ingest.mapillary_sequences import fetch_roi
from pose_alignment.geo_common import RoiPaths
from pose_alignment.route_gui import make_roi

LON0, LAT0 = -83.7400, 42.2800
M_LON = 1 / (111320.0 * np.cos(np.radians(LAT0)))  # degrees of longitude per metre


def _record(k, lon, lat=LAT0, seq="seqAAAAAAAA"):
    return {
        "id": str(1000 + k),
        "sequence": seq,
        "captured_at": 1_700_000_000_000 + 1000 * k,
        "camera_type": "perspective",
        "camera_parameters": [0.8, 0.0, 0.0],
        "make": "Apple",
        "model": "iPhone",
        "width": 2400,
        "height": 1800,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "computed_geometry": {"type": "Point", "coordinates": [lon, lat]},
        "compass_angle": 90.0,
        "computed_compass_angle": 90.0,
        "computed_rotation": [1.2, -1.2, 1.2],
        "thumb_2048_url": f"https://example.com/2048/{k}.jpg",
        "thumb_original_url": f"https://example.com/original/{k}.jpg",
    }


@pytest.fixture
def photo_bytes():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (1800, 2400, 3), np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_image_store_downloads_each_photo_once(tmp_path, monkeypatch, photo_bytes):
    calls = []
    monkeypatch.setattr(
        image_store,
        "fetch_bytes",
        lambda url, timeout=120: calls.append(url) or photo_bytes,
    )
    store = ImageStore(tmp_path / "originals")
    assert store.get("1", "https://example.com/1.jpg") == photo_bytes
    assert store.get("1", "https://example.com/1.jpg") == photo_bytes
    assert calls == ["https://example.com/1.jpg"] and store.has("1")
    monkeypatch.setattr(image_store, "fetch_bytes", lambda url, timeout=120: None)
    assert store.get("2", "https://example.com/2.jpg") is None and not store.has("2")


def test_fetch_roi_picks_a_run_from_given_records_and_scales_stored_originals(
    tmp_path, monkeypatch, photo_bytes
):
    monkeypatch.setattr(
        image_store, "fetch_bytes", lambda url, timeout=120: photo_bytes
    )
    # 30 frames 5 m apart along a 150 m street, plus one image of another sequence far away
    records = [_record(k, LON0 + 5 * k * M_LON) for k in range(30)]
    records.append(_record(99, LON0, LAT0 + 0.01, seq="seqFARAWAY0"))
    roi = make_roi("test", [[(LON0, LAT0), (LON0 + 145 * M_LON, LAT0)]], 15.0)

    out = fetch_roi(
        roi, images=records, keep_originals=True, out=tmp_path / "roi", no_rig=True
    )

    paths = RoiPaths(out)
    (seq,) = paths.sequences()
    assert seq["folder"] == "line1_cam0_seqAAAAA" and seq["frames"] == 30
    frames = paths.frames(seq["folder"])
    assert frames["frame_0001.jpg"]["id"] == "1000"
    frame = cv2.imread(str(paths.images / seq["folder"] / "frame_0001.jpg"))
    assert frame.shape[:2] == (1536, 2048)  # scaled from the stored 2400 px original
    assert ImageStore(paths.originals).has("1000")
    assert json.loads(paths.roi_json.read_text())["lines"]


class _State:
    def __init__(self, meta):
        self.meta = meta

    def get_metadata(self, key, default=None):
        return self.meta.get(key, default)

    def require_metadata(self, key):
        return self.meta[key]

    def set_metadata(self, key, value):
        self.meta[key] = value


class _Progress:
    def next(self):
        pass


def _line(m0, m1):
    return [(LON0 + m0 * M_LON, LAT0), (LON0 + m1 * M_LON, LAT0)]


def test_chooser_adds_a_run_where_the_best_one_ends():
    from data_ingest.mapillary_sequences import choose_roi, selection_images

    # a 300 m street: one capture along its first half, another along its second half,
    # and a third, sparser one along the first half again
    first = [_record(k, LON0 + 5 * k * M_LON, seq="seqFIRST000") for k in range(30)]
    second = [
        _record(100 + k, LON0 + (155 + 5 * k) * M_LON, seq="seqSECOND00")
        for k in range(30)
    ]
    again = [
        _record(200 + k, LON0 + 6 * k * M_LON, seq="seqAGAIN000") for k in range(24)
    ]
    roi = make_roi("test", [_line(0, 300)], 15.0)

    selection = choose_roi(roi, first + second + again, no_rig=True)
    picks = [g["pick"]["sequence"] for g in selection["groups"]]
    assert sorted(picks) == ["seqFIRST000", "seqSECOND00"]
    assert {g["street_line"] for g in selection["groups"]} == {1}
    assert len(selection_images(selection)) == 60
    assert (
        len(
            choose_roi(roi, first + second + again, no_rig=True, one_run=True)["groups"]
        )
        == 1
    )


def test_ingest_textures_only_the_chosen_runs(monkeypatch):
    from pose_alignment import streets_from_bbox

    bbox = BoundingBox(LAT0 - 0.001, LAT0 + 0.002, LON0 - 0.001, LON0 + 0.003)
    run = [_record(k, LON0 + 5 * k * M_LON) for k in range(30)]
    stray = [
        _record(500 + k, LON0 + 0.0005, LAT0 + 0.0015, seq=f"seqSTRAY{k:03d}")
        for k in range(3)
    ]
    asked = {}

    def fetch_images(self, box, fields, limit=2000, tile_deg=0.0005):
        asked["box"], asked["fields"] = box, fields
        return run + stray

    monkeypatch.setattr(ingest.MapillaryClient, "fetch_images", fetch_images)
    monkeypatch.setattr(
        ingest.OSMClient, "fetch", lambda self, box, features: {"elements": []}
    )
    monkeypatch.setattr(
        streets_from_bbox,
        "streets_roi",
        lambda box: make_roi("box", [_line(0, 145)], 15.0),
    )

    data = ingest.ingest_data(bbox, _State({"progress_monitor": _Progress()}))
    assert set(data["mapillary"]) == {r["id"] for r in run}  # the strays are not chosen
    assert data["mapillary_selection"]["groups"][0]["pick"]["sequence"] == "seqAAAAAAAA"
    assert (
        asked["box"].max_lon > bbox.max_lon + 0.0008
        and "captured_at" in asked["fields"]
    )

    # no street (e.g. Overpass down): every image in the box, as before
    def no_streets(box):
        raise RuntimeError("Overpass is down")

    monkeypatch.setattr(streets_from_bbox, "streets_roi", no_streets)
    with pytest.warns(UserWarning, match="Choosing captures failed"):
        data = ingest.ingest_data(bbox, _State({"progress_monitor": _Progress()}))
    assert len(data["mapillary"]) == 33 and "mapillary_selection" not in data


def test_a_360_run_beats_a_forward_camera_covering_the_street_as_well():
    from data_ingest.mapillary_sequences import rank_runs

    def run(seq, pano, view, images=40):
        return {
            "sequence": seq,
            "pano": pano,
            "view_deg": view,
            "straight": images,
            "images": images,
            "spacing_m": 3.1,
            "coverage": 1.0,
        }

    runs = [
        run("forward", False, 2.0, 106),
        run("pano", True, -97.0),
        run("side", False, 88.0),
    ]
    assert [r["sequence"] for r in rank_runs(runs)] == ["pano", "side", "forward"]
    assert [r["sequence"] for r in rank_runs(runs, pano="off")] == ["side", "forward"]
    # coverage still comes first: a forward camera covering more of the street wins
    runs[0]["coverage"] = 1.0
    runs[1]["coverage"] = runs[2]["coverage"] = 0.5
    assert rank_runs(runs)[0]["sequence"] == "forward"


def test_one_capture_chosen_on_two_streets_becomes_one_run():
    from data_ingest.mapillary_sequences import choose_roi, selection_images

    M_LAT = 1 / 111320.0
    # one drive: 30 frames east along street 1, then 30 frames north along street 2
    east = [_record(k, LON0 + 5 * k * M_LON) for k in range(30)]
    north = [
        _record(30 + k, LON0 + 150 * M_LON, LAT0 + 5 * (k + 1) * M_LAT)
        for k in range(30)
    ]
    roi = make_roi(
        "L",
        [
            _line(0, 145),
            [
                (LON0 + 150 * M_LON, LAT0 + 5 * M_LAT),
                (LON0 + 150 * M_LON, LAT0 + 150 * M_LAT),
            ],
        ],
        15.0,
    )
    selection = choose_roi(roi, east + north, no_rig=True)
    (group,) = selection["groups"]
    assert group["street_lines"] == [1, 2] and group["line"] == 1
    run_ids = [i["id"] for i in group["pick"]["run"]]
    assert run_ids == sorted(run_ids, key=int) and len(run_ids) == 60
    assert len(selection_images(selection)) == 60


def test_separate_passes_of_one_sequence_stay_separate():
    from data_ingest.mapillary_sequences import merge_repeated

    def group(k, ids, t0):
        run = [
            {"id": str(i), "sequence": "s", "captured_at": t0 + 1000 * n}
            for n, i in enumerate(ids)
        ]
        return {
            "line": k,
            "street_line": k,
            "pick": {"sequence": "s", "pano": False, "run": run},
            "siblings": [],
        }

    # the second pass along the other street is 10 minutes later
    out = merge_repeated([group(1, range(5), 0), group(2, range(5, 10), 600_000)], {})
    assert [g["line"] for g in out] == [1, 2]


def test_buildings_for_the_mesh_and_for_the_alignment():
    from pose_alignment.osm import buildings_from_elements

    def way(lon, lat, d=0.0001):
        ring = [
            (lon, lat),
            (lon + d, lat),
            (lon + d, lat + d),
            (lon, lat + d),
            (lon, lat),
        ]
        return {
            "type": "way",
            "tags": {"building": "yes"},
            "geometry": [{"lon": x, "lat": y} for x, y in ring],
        }

    bbox = BoundingBox(LAT0, LAT0 + 0.001, LON0, LON0 + 0.001)
    inside_box, outside_box = way(LON0 + 0.0005, LAT0 + 0.0005), way(LON0 + 0.002, LAT0)
    node = {"type": "node", "tags": {"building": "yes"}, "lat": LAT0, "lon": LON0}
    assert ingest.overlaps(inside_box, bbox) and not ingest.overlaps(outside_box, bbox)
    buildings = buildings_from_elements([inside_box, outside_box, node])
    assert (
        len(buildings) == 2
        and buildings[0][0] == {"building": "yes"}
        and len(buildings[0][1]) == 5
    )
