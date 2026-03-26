from __future__ import annotations

import json
import os
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch

def _gpu_available() -> bool:
    return torch.cuda.is_available()


def _gpu_snapshot() -> dict[str, Any]:
    if not _gpu_available():
        return {
            "allocated_mb": "N/A",
            "reserved_mb": "N/A",
            "free_mb": "N/A",
        }
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 2),
        "free_mb": round(
            (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e6,
            2,
        ),
    }


def _gpu_system_info() -> dict[str, Any]:
    if not _gpu_available():
        return {"available": False, "name": "N/A", "total_vram_mb": "N/A", "count": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_vram_mb": round(props.total_memory / 1e6, 1),
        "count": torch.cuda.device_count(),
    }


@dataclass
class StageMetrics:
    name: str
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    start_memory_mb: float = 0.0
    end_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    gpu_start: dict[str, Any] = field(default_factory=dict)
    gpu_end: dict[str, Any] = field(default_factory=dict)
    gpu_delta_allocated_mb: Any = "N/A"
    throughput: dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    error: str | None = None

    def _start(self) -> None:
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()
        self.start_memory_mb = sum(s.size for s in snapshot.statistics("lineno")) / 1e6
        self.gpu_start = _gpu_snapshot()

    def _stop(self, status: str = "ok", error: str | None = None) -> None:
        self.wall_time_s = time.perf_counter() - self._wall_start
        self.cpu_time_s = time.process_time() - self._cpu_start
        snapshot = tracemalloc.take_snapshot()
        self.end_memory_mb = sum(s.size for s in snapshot.statistics("lineno")) / 1e6
        _, peak = tracemalloc.get_traced_memory()
        self.peak_memory_mb = peak / 1e6
        self.memory_delta_mb = self.end_memory_mb - self.start_memory_mb
        tracemalloc.stop()
        self.gpu_end = _gpu_snapshot()
        if _gpu_available():
            self.gpu_delta_allocated_mb = round(
                float(self.gpu_end["allocated_mb"]) - float(self.gpu_start["allocated_mb"]), 2
            )
        self.status = status
        self.error = error

    def set_throughput(self, label: str, value: float) -> None:
        self.throughput[label] = value


@dataclass
class PipelineProfiler:
    pipeline_name: str
    _stages: list[StageMetrics] = field(default_factory=list, init=False)
    _started_at: str = field(default="", init=False)
    _finished_at: str = field(default="", init=False)
    _wall_start: float = field(default=0.0, init=False)

    def start(self) -> None:
        self._wall_start = time.perf_counter()
        self._started_at = datetime.now(timezone.utc).isoformat()

    def begin_stage(self, name: str) -> StageMetrics:
        metrics = StageMetrics(name=name)
        metrics._start()
        self._stages.append(metrics)
        return metrics

    def end_stage(
        self, metrics: StageMetrics, status: str = "ok", error: str | None = None
    ) -> None:
        metrics._stop(status=status, error=error)

    def finish(self) -> None:
        self._finished_at = datetime.now(timezone.utc).isoformat()

    @property
    def total_wall_time_s(self) -> float:
        return time.perf_counter() - self._wall_start

    def summary(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline_name,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "total_wall_time_s": round(self.total_wall_time_s, 3),
            "system": {
                "python": platform.python_version(),
                "os": platform.platform(),
                "cpu_count": os.cpu_count(),
                "gpu": _gpu_system_info(),
            },
            "stages": [
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in asdict(s).items()
                    if not k.startswith("_")
                }
                for s in self._stages
            ],
        }

    def save(self, path: str = "performance.json") -> None:
        data = self.summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Performance report saved -> {path}")
        self._print_table()

    def _print_table(self) -> None:
        has_gpu = _gpu_available()
        header = f"{'Stage':<20} {'Wall(s)':>8} {'CPU(s)':>8} {'Peak MB':>9}"
        header += f" {'GPU Alloc MB':>13}" if has_gpu else f" {'GPU':>13}"
        header += f" {'Status':>8}"
        print(f"\n{header}")
        print("-" * (58 + 14))
        for s in self._stages:
            gpu_col = str(s.gpu_delta_allocated_mb)
            print(
                f"{s.name:<20} {s.wall_time_s:>8.2f} {s.cpu_time_s:>8.2f}"
                f" {s.peak_memory_mb:>9.1f} {gpu_col:>13} {s.status:>8}"
            )
        print("-" * (58 + 14))
        print(f"{'TOTAL':<20} {self.total_wall_time_s:>8.2f}")
