import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import geocode
from api.app import create_app
from common import PipelineChain

BBOX = {"west": -83.71978, "south": 42.29025, "east": -83.71205, "north": 42.29422}


def fetch(value, state):
    return {
        "osm": {"elements": [{"id": 1, "geometry": []}, {"id": 2}]},
        "mapillary": {"img1": {}, "img2": {}},
    }


def export(value, state):
    Path(state.metadata["output_path"]).write_bytes(b"glTF-binary")
    images = Path(state.metadata["image_dir"])
    images.mkdir(parents=True, exist_ok=True)
    (images / "img1_original.jpg").write_bytes(b"jpeg")
    (images / "images.json").write_text(
        json.dumps(
            [
                {
                    "id": "img1",
                    "heading": 90,
                    "original": "img1_original.jpg",
                    "cleaned": None,
                }
            ]
        )
    )
    return value


def fake_pipeline():
    pipeline = PipelineChain("fake")
    pipeline.add_stage("fetch", fetch, label="Fetch data", detail="OSM + Mapillary")
    pipeline.add_stage("export", export, label="Export", detail="Write .glb")
    return pipeline


def failing_pipeline():
    def no_images(value, state):
        raise RuntimeError("No images found.")

    pipeline = PipelineChain("failing")
    pipeline.add_stage("fetch", no_images, label="Fetch data")
    pipeline.add_stage("export", export, label="Export")
    return pipeline


def make_client(tmp_path, factory=fake_pipeline):
    return TestClient(
        create_app(data_dir=tmp_path, pipeline_factory=factory, serve_web=False)
    )


def wait_for(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_pipeline_endpoint_lists_stages(tmp_path):
    client = make_client(tmp_path)

    stages = client.get("/api/pipeline").json()["stages"]

    assert [s["name"] for s in stages] == ["fetch", "export"]
    assert stages[0]["label"] == "Fetch data"


def test_job_runs_and_serves_model_and_images(tmp_path):
    client = make_client(tmp_path)

    created = client.post("/api/jobs", json={"bbox": BBOX, "name": "Test block"})
    assert created.status_code == 201
    job = wait_for(client, created.json()["id"])

    assert job["status"] == "done"
    assert job["stages"] == {"fetch": "done", "export": "done"}
    assert job["result"]["buildings"] == 1
    assert job["result"]["photos"] == 2
    assert job["result"]["thumbnail"] == "img1_original.jpg"

    model = client.get(f"/api/jobs/{job['id']}/model.glb?download=true")
    assert model.status_code == 200
    assert model.content == b"glTF-binary"
    assert "posm-test-block" in model.headers["content-disposition"]

    images = client.get(f"/api/jobs/{job['id']}/images").json()["images"]
    assert images[0]["original"] == f"/api/jobs/{job['id']}/images/img1_original.jpg"
    assert client.get(images[0]["original"]).content == b"jpeg"


def test_failed_job_reports_error_and_stage(tmp_path):
    client = make_client(tmp_path, failing_pipeline)

    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    job = wait_for(client, job_id)

    assert job["status"] == "failed"
    assert job["error"] == "No images found."
    assert job["stages"] == {"fetch": "error", "export": "pending"}
    assert client.get(f"/api/jobs/{job_id}/model.glb").status_code == 404


def test_finished_jobs_are_reloaded_after_restart(tmp_path):
    client = make_client(tmp_path)
    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    wait_for(client, job_id)

    restarted = make_client(tmp_path)

    jobs = restarted.get("/api/jobs").json()["jobs"]
    assert [j["id"] for j in jobs] == [job_id]
    assert jobs[0]["status"] == "done"


@pytest.mark.parametrize(
    "bbox, message",
    [
        ({**BBOX, "west": BBOX["east"] + 0.01}, "West must be less than east"),
        ({**BBOX, "east": BBOX["west"] + 0.1}, "Keep each side under"),
    ],
)
def test_rejects_bad_bounding_boxes(tmp_path, bbox, message):
    client = make_client(tmp_path)

    resp = client.post("/api/jobs", json={"bbox": bbox})

    assert resp.status_code == 422
    assert message in json.dumps(resp.json())


def test_image_names_cannot_escape_job_folder(tmp_path):
    client = make_client(tmp_path)
    job_id = client.post("/api/jobs", json={"bbox": BBOX}).json()["id"]
    wait_for(client, job_id)

    assert client.get(f"/api/jobs/{job_id}/images/..%2Fjob.json").status_code == 404


def test_geocode_parses_and_caches_results(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "name": "Pierpont Commons",
                    "display_name": "Pierpont Commons, Bonisteel Blvd, Ann Arbor",
                    "lat": "42.2913",
                    "lon": "-83.7173",
                    "boundingbox": ["42.2908", "42.2918", "-83.7180", "-83.7166"],
                }
            ]

    def fake_get(url, **kwargs):
        calls.append(kwargs["params"]["q"])
        return FakeResponse()

    monkeypatch.setattr(geocode.requests, "get", fake_get)
    monkeypatch.setattr(geocode, "_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(geocode, "_cache", type(geocode._cache)())
    client = make_client(tmp_path)

    first = client.get("/api/geocode", params={"q": "Pierpont Commons"}).json()
    client.get("/api/geocode", params={"q": "pierpont commons "})

    place = first["results"][0]
    assert place["name"] == "Pierpont Commons"
    assert place["bbox"] == {
        "west": -83.7180,
        "south": 42.2908,
        "east": -83.7166,
        "north": 42.2918,
    }
    assert calls == ["Pierpont Commons"]


def test_real_pipeline_stages_drive_the_timeline():
    from pipeline import build_pipeline

    assert [s["name"] for s in build_pipeline().describe()] == [
        "fetch",
        "build_mesh",
        "texturing",
        "export",
    ]


def test_suggest_uses_photon_and_handles_points_and_extents(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "features": [
                    {
                        "geometry": {"coordinates": [-83.7176, 42.2911]},
                        "properties": {
                            "name": "Pierpont Commons",
                            "city": "Ann Arbor",
                            "state": "Michigan",
                            "country": "United States",
                            "extent": [-83.7180, 42.2916, -83.7170, 42.2907],
                        },
                    },
                    {
                        "geometry": {"coordinates": [-83.7483, 42.2808]},
                        "properties": {
                            "housenumber": "300",
                            "street": "South Main Street",
                            "city": "Ann Arbor",
                        },
                    },
                ]
            }

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return FakeResponse()

    monkeypatch.setattr(geocode.requests, "get", fake_get)
    monkeypatch.setattr(geocode, "_suggest_cache", type(geocode._suggest_cache)())
    client = make_client(tmp_path)

    resp = client.get(
        "/api/suggest", params={"q": "pierp", "lat": 42.2911, "lon": -83.7176}
    )
    client.get("/api/suggest", params={"q": "pierp", "lat": 42.2912, "lon": -83.7177})

    first, second = resp.json()["results"]
    assert first["address"] == "Pierpont Commons, Ann Arbor, Michigan, United States"
    assert first["bbox"] == {
        "west": -83.7180,
        "south": 42.2907,
        "east": -83.7170,
        "north": 42.2916,
    }
    assert second["name"] == "300 South Main Street"
    assert second["bbox"]["west"] == second["bbox"]["east"] == -83.7483
    assert calls[0][0] == geocode.PHOTON_URL
    assert calls[0][1]["lat"] == 42.29
    assert len(calls) == 1  # nearby bias point reuses the cached answer


def test_suggest_ignores_short_queries(tmp_path):
    client = make_client(tmp_path)

    assert client.get("/api/suggest", params={"q": "ab"}).json() == {"results": []}
