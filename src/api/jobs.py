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
from common import PipelineChain, PipelineProfiler, ProgressMonitor
from common.constants import BoundingBox
from common.redact import redact

# Stages that only do something when their job option is on.
OPTIONAL_STAGES = {"terrain": "terrain", "naip": "naip"}

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
    # With the "profile" option: per-stage time and memory, updated after each stage
    # while the job runs (profiler.public_summary), so the page can show it live.
    profile: dict[str, Any] | None = None

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
            stages={
                s["name"]: "skipped" if _turned_off(s["name"], options) else "pending"
                for s in self.stages
            },
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
        # This is the only worker, so nothing a single job does (e.g. a disk
        # error writing its log or job.json) may stop it: later jobs would
        # stay queued forever.
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
                if job is not None:
                    self._run(job)
            except Exception as exc:
                print(f"[{job_id}] Job crashed outside the pipeline:", flush=True)
                print(redact(traceback.format_exc()), flush=True)
                self._mark_crashed(job_id, exc)
            finally:
                self._queue.task_done()

    def _mark_crashed(self, job_id: str, exc: Exception) -> None:
        """Best effort: record the failure; the job may not even be writable."""
        job = self.get(job_id)
        if job is None or job.status in ("done", "failed"):
            return
        try:
            self._update(
                job,
                status="failed",
                error=_describe_error(exc),
                finished_at=_now(),
                stages={
                    k: ("error" if v == "running" else v) for k, v in job.stages.items()
                },
            )
        except Exception:
            # Can't save it; at least tell connected browsers.
            job.status = "failed"
            if self._publish:
                self._publish({"type": "job", "job": job.to_dict()})

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
            if profiler is not None:
                job.profile = profiler.public_summary(
                    running=name if event == "start" else None
                )
                if event != "start":  # the download is current too, mid-run
                    profiler.write(str(job_dir / "profile.json"))
            if job.stages.get(name) == "skipped":
                # The stage runs but does nothing for this job; only the profile changed.
                if profiler is not None:
                    with self._lock:
                        self._save(job)
                return
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
        profiler = (
            PipelineProfiler(pipeline_name="photorealistic-street-maps")
            if opts.get("profile")
            else None
        )
        skip_stages = {
            name for name, status in job.stages.items() if status == "skipped"
        }
        try:
            state = self._pipeline_factory().run(
                bbox,
                metadata={
                    "bbox": bbox,
                    "progress_monitor": ProgressMonitor(verbose=False),
                    "remove_obstructions": opts.get("remove_obstructions", True),
                    "seg_model": SEG_MODELS.get(opts.get("seg_model", "large")),
                    "seg_device": opts.get("device", "auto"),
                    "use_terrain": opts.get("terrain", False),
                    "use_naip": opts.get("naip", False),
                    "output_path": str(model_path),
                    "image_dir": str(job_dir / "images"),
                    "reporter": log,
                },
                profiler=profiler,
                on_stage=on_stage,
                skip_stages=skip_stages,
            )
        except Exception as exc:
            outcome: dict[str, Any] = {
                "status": "failed",
                "error": _describe_error(exc),
            }
            if profiler is not None:
                profiler.finish()  # run() only finishes it when every stage succeeds
                outcome["profile"] = _write_profile(profiler, job_dir, log)
            log.error(traceback.format_exc())
            return outcome

        result = _summarize(state, job_dir)
        outcome = {"status": "done", "result": result}
        if profiler is not None:
            result["profile"] = outcome["profile"] = _write_profile(
                profiler, job_dir, log
            )
        return outcome


def _write_profile(
    profiler: PipelineProfiler, job_dir: Path, log: JobReporter
) -> dict[str, Any]:
    profiler.save(str(job_dir / "profile.json"), reporter=log)
    return profiler.public_summary()


def _turned_off(stage: str, options: dict[str, Any]) -> bool:
    """Whether an optional stage is off for a job (shown as skipped)."""
    option = OPTIONAL_STAGES.get(stage)
    return option is not None and not options.get(option, False)


def _describe_error(exc: Exception) -> str:
    msg = redact(str(exc)).strip() or type(exc).__name__
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
