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


def test_ingest_keeps_the_alignments_records_only_when_aligning(monkeypatch):
    bbox = BoundingBox(LAT0 - 0.001, LAT0 + 0.001, LON0 - 0.001, LON0 + 0.001)
    records = [
        _record(0, LON0),
        _record(1, LON0 + 0.0015),
    ]  # the second is past the box
    asked = {}

    def fetch_images(self, box, fields, limit=2000, tile_deg=0.0005):
        asked["box"], asked["fields"] = box, fields
        return records

    monkeypatch.setattr(ingest.MapillaryClient, "fetch_images", fetch_images)
    monkeypatch.setattr(
        ingest.OSMClient, "fetch", lambda self, box, features: {"elements": []}
    )

    class State:
        def __init__(self, meta):
            self.meta = meta

        def get_metadata(self, key, default=None):
            return self.meta.get(key, default)

        def require_metadata(self, key):
            return self.meta[key]

        def set_metadata(self, key, value):
            self.meta[key] = value

    class Progress:
        def next(self):
            pass

    data = ingest.ingest_data(
        bbox, State({"progress_monitor": Progress(), "pose_source": "colmap"})
    )
    assert list(data["mapillary"]) == ["1000"]  # texturing: the box only
    assert len(data["mapillary_images"]) == 2  # the alignment: the grown box
    assert asked["box"].max_lon > bbox.max_lon + 0.0008 and "make" in asked["fields"]

    data = ingest.ingest_data(bbox, State({"progress_monitor": Progress()}))
    assert "mapillary_images" not in data and "make" not in asked["fields"]
