"""Background job runner for the web API.

Jobs run one at a time on a single worker thread: the segmentation and
inpainting models are large, so running several pipelines at once would just
compete for the same CPU/GPU memory. Each job keeps its files in
<data_dir>/<job_id>/ (job.json, pipeline.log, model.glb, images/), so finished
jobs survive a restart.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.joblog import JobReporter
from common import PipelineChain, ProgressMonitor
from common.constants import BoundingBox

SEG_MODELS = {
    "large": "facebook/mask2former-swin-large-cityscapes-semantic",
    "tiny": "facebook/mask2former-swin-tiny-cityscapes-semantic",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    name: str
    bbox: dict[str, float]  # west, south, east, north
    options: dict[str, Any]
    stages: dict[str, str]  # stage name -> pending | running | done | error
    status: str = "queued"  # queued | running | done | failed
    error: str | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    def __init__(
        self,
        data_dir: Path,
        pipeline_factory: Callable[[], PipelineChain],
        publish: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """publish receives live messages for browsers:
        {"type": "job", "job": {...}} after every job change, and
        {"type": "log" | "progress", "job_id": ..., ...} for pipeline output.
        It's called from the worker thread, so it must be quick and thread-safe."""
        self.data_dir = Path(data_dir)
        self._publish = publish
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline_factory = pipeline_factory
        self.stages = pipeline_factory().describe()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._load_existing()
        self._worker = threading.Thread(target=self._run_forever, daemon=True)
        self._worker.start()

    # ---- public API ----------------------------------------------------

    def submit(self, name: str, bbox: dict[str, float], options: dict[str, Any]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            name=name,
            bbox=bbox,
            options=options,
            stages={s["name"]: "pending" for s in self.stages},
        )
        with self._lock:
            self._jobs[job.id] = job
            self._save(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    # ---- internals -----------------------------------------------------

    def _save(self, job: Job) -> None:
        d = self.job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        data = job.to_dict()
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(d / "job.json")
        if self._publish:
            self._publish({"type": "job", "job": data})

    def _update(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for k, v in changes.items():
                setattr(job, k, v)
            self._save(job)

    def _load_existing(self) -> None:
        for path in self.data_dir.glob("*/job.json"):
            try:
                job = Job(**json.loads(path.read_text()))
            except (OSError, TypeError, ValueError):
                continue
            if job.status in ("queued", "running"):
                # The process stopped mid-run; that work is lost.
                job.status = "failed"
                job.error = "The server restarted before this job finished."
                job.stages = {
                    k: ("error" if v == "running" else v) for k, v in job.stages.items()
                }
                self._save(job)
            self._jobs[job.id] = job

    def _run_forever(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is not None:
                self._run(job)
            self._queue.task_done()

    def _run(self, job: Job) -> None:
        job_dir = self.job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        model_path = job_dir / "model.glb"

        def emit(message: dict[str, Any]) -> None:
            if self._publish:
                self._publish({**message, "job_id": job.id})

        log = JobReporter(job_dir / "pipeline.log", emit, prefix=f"[{job.id}] ")
        started = time.monotonic()
        self._update(job, status="running", started_at=_now())
        # Finish and close the log before the final status goes out, so a page
        # that reloads the log when the job ends gets every line.
        try:
            outcome = self._run_pipeline(job, job_dir, model_path, log)
            elapsed = time.monotonic() - started
            if outcome["status"] == "done":
                log.info(f"Done in {elapsed:.0f}s.")
            else:
                log.info(f"Failed after {elapsed:.0f}s: {outcome['error']}")
        finally:
            log.close()
        self._update(job, finished_at=_now(), **outcome)

    def _run_pipeline(
        self, job: Job, job_dir: Path, model_path: Path, log: JobReporter
    ) -> dict[str, Any]:
        """Run the pipeline and return the job's final status fields."""
        b = job.bbox
        log.info(
            f"Job {job.id} · {job.name} · bbox W {b['west']:.5f} S {b['south']:.5f} "
            f"E {b['east']:.5f} N {b['north']:.5f}"
        )
        log.info("Options: " + ", ".join(f"{k}={v}" for k, v in job.options.items()))
        labels = {s["name"]: s["label"] for s in self.stages}

        def on_stage(name: str, event: str) -> None:
            if event == "start":
                log.info(f"── {labels.get(name, name)} ──")
            state = {"start": "running", "done": "done", "error": "error"}[event]
            with self._lock:
                if name in job.stages:
                    job.stages[name] = state
                self._save(job)

        bbox = BoundingBox(
            min_lat=b["south"], max_lat=b["north"], min_lon=b["west"], max_lon=b["east"]
        )
        opts = job.options
        try:
            state = self._pipeline_factory().run(
                bbox,
                metadata={
                    "bbox": bbox,
                    "progress_monitor": ProgressMonitor(verbose=False),
                    "remove_obstructions": opts.get("remove_obstructions", True),
                    "seg_model": SEG_MODELS.get(opts.get("seg_model", "large")),
                    "seg_device": opts.get("device", "auto"),
                    "output_path": str(model_path),
                    "image_dir": str(job_dir / "images"),
                    "reporter": log,
                },
                on_stage=on_stage,
            )
        except Exception as exc:
            log.error(traceback.format_exc())
            return {"status": "failed", "error": _describe_error(exc)}

        return {"status": "done", "result": _summarize(state, job_dir)}


def _describe_error(exc: Exception) -> str:
    msg = str(exc).strip() or type(exc).__name__
    return msg if len(msg) <= 300 else msg[:297] + "..."


def _summarize(state: Any, job_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fetched = state.stage_outputs.get("fetch") or {}
    elements = (fetched.get("osm") or {}).get("elements", [])
    result["buildings"] = sum(1 for e in elements if "geometry" in e)
    result["photos"] = len(fetched.get("mapillary") or {})
    mesh = state.stage_outputs.get("texturing")
    faces = getattr(mesh, "faces", None)
    if faces is not None:
        result["faces"] = int(len(faces))
    model_path = job_dir / "model.glb"
    if model_path.exists():
        result["model_bytes"] = model_path.stat().st_size
    index = job_dir / "images" / "images.json"
    if index.exists():
        try:
            images = json.loads(index.read_text())
        except ValueError:
            images = []
        if images:
            result["thumbnail"] = images[0].get("cleaned") or images[0]["original"]
    return result
