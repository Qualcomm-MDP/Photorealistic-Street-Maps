"""Web API and static frontend for the street-map pipeline.

Run locally:   poetry run uvicorn api.app:create_app --factory --app-dir src
With Docker:   docker compose up api   (then open http://localhost:8000)
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from api import geocode
from api.jobs import SEG_MODELS, JobManager
from common import PipelineChain
from common.env import load_app_env

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
DEFAULT_DATA_DIR = REPO_ROOT / "output" / "jobs"

# Large areas mean thousands of street photos and Overpass/Mapillary timeouts.
MAX_SIDE_M = 2500

_IMAGE_NAME = re.compile(r"^[\w-]+\.jpg$")


class BBox(BaseModel):
    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def check_shape(self) -> "BBox":
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("West must be less than east, and south less than north.")
        lat = math.radians((self.south + self.north) / 2)
        width = (self.east - self.west) * 111_320 * math.cos(lat)
        height = (self.north - self.south) * 110_574
        if max(width, height) > MAX_SIDE_M:
            raise ValueError(
                f"Area is {width:.0f} × {height:.0f} m. "
                f"Keep each side under {MAX_SIDE_M} m."
            )
        return self


class JobRequest(BaseModel):
    bbox: BBox
    name: str = Field(default="Custom area", max_length=120)
    remove_obstructions: bool = True
    seg_model: Literal["large", "tiny"] = "large"
    device: Literal["auto", "cpu", "cuda"] = "auto"


def _default_pipeline() -> PipelineChain:
    # Imported here so creating an app with a custom factory (e.g. in tests)
    # doesn't load torch and the models' dependencies.
    from pipeline import build_pipeline

    return build_pipeline()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def create_app(
    data_dir: Path | None = None,
    pipeline_factory: Callable[[], PipelineChain] = _default_pipeline,
    serve_web: bool = True,
) -> FastAPI:
    load_app_env()
    data_dir = Path(data_dir or os.environ.get("POSM_DATA_DIR") or DEFAULT_DATA_DIR)
    jobs = JobManager(data_dir, pipeline_factory)
    app = FastAPI(title="POSM", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.jobs = jobs

    def _job_or_404(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    @app.get("/api/pipeline")
    def pipeline_stages() -> dict[str, Any]:
        return {"stages": jobs.stages}

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return {
            "cuda": _cuda_available(),
            "seg_models": list(SEG_MODELS),
            "max_side_m": MAX_SIDE_M,
        }

    @app.get("/api/geocode")
    def search_address(q: str) -> dict[str, Any]:
        q = q.strip()
        if len(q) < 3:
            raise HTTPException(422, "Type at least 3 characters.")
        try:
            return {"results": geocode.search(q)}
        except requests.RequestException as exc:
            raise HTTPException(502, f"Address search is unavailable: {exc}") from exc

    @app.get("/api/suggest")
    def suggest_address(
        q: str, lat: float | None = None, lon: float | None = None
    ) -> dict[str, Any]:
        q = q.strip()
        if len(q) < 3:
            return {"results": []}
        try:
            return {"results": geocode.suggest(q, lat, lon)}
        except requests.RequestException as exc:
            raise HTTPException(502, f"Suggestions are unavailable: {exc}") from exc

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": [j.to_dict() for j in jobs.list_jobs()]}

    @app.post("/api/jobs", status_code=201)
    def create_job(req: JobRequest) -> dict[str, Any]:
        if req.device == "cuda" and not _cuda_available():
            raise HTTPException(422, "No CUDA GPU is available on this server.")
        job = jobs.submit(
            name=req.name.strip() or "Custom area",
            bbox=req.bbox.model_dump(),
            options={
                "remove_obstructions": req.remove_obstructions,
                "seg_model": req.seg_model,
                "device": req.device,
            },
        )
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _job_or_404(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/model.glb")
    def get_model(job_id: str, download: bool = False) -> FileResponse:
        job = _job_or_404(job_id)
        path = jobs.job_dir(job.id) / "model.glb"
        if job.status != "done" or not path.exists():
            raise HTTPException(404, "This job has no model yet.")
        slug = re.sub(r"[^a-z0-9]+", "-", job.name.lower()).strip("-") or "model"
        return FileResponse(
            path,
            media_type="model/gltf-binary",
            filename=f"posm-{slug}-{job.id}.glb" if download else None,
        )

    @app.get("/api/jobs/{job_id}/images")
    def list_images(job_id: str) -> dict[str, Any]:
        job = _job_or_404(job_id)
        index = jobs.job_dir(job.id) / "images" / "images.json"
        if not index.exists():
            return {"images": []}
        base = f"/api/jobs/{job.id}/images"
        images = []
        for item in json.loads(index.read_text()):
            images.append(
                {
                    "id": item["id"],
                    "heading": item.get("heading"),
                    "original": f"{base}/{item['original']}",
                    "cleaned": f"{base}/{item['cleaned']}"
                    if item.get("cleaned")
                    else None,
                }
            )
        return {"images": images}

    @app.get("/api/jobs/{job_id}/images/{name}")
    def get_image(job_id: str, name: str) -> FileResponse:
        job = _job_or_404(job_id)
        if not _IMAGE_NAME.match(name):
            raise HTTPException(404, "Image not found.")
        path = jobs.job_dir(job.id) / "images" / name
        if not path.exists():
            raise HTTPException(404, "Image not found.")
        return FileResponse(path, media_type="image/jpeg")

    if serve_web and WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app
