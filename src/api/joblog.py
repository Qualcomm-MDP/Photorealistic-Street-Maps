"""Per-job pipeline logs for the web API.

JobReporter is the Reporter the API passes to the pipeline (see
common/reporter.py). Every line goes to three places: the job's pipeline.log,
the browser (through the emit callback, over the WebSocket), and the server
console so `docker compose logs` still shows it. Progress bars are streamed as
they move, throttled, and only their final state is saved to the file.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from common.reporter import Reporter

T = TypeVar("T")


class JobReporter(Reporter):
    def __init__(
        self,
        path: Path,
        emit: Callable[[dict[str, Any]], None],
        prefix: str = "",
        progress_interval_s: float = 0.25,
    ) -> None:
        self.path = Path(path)
        self._file = open(self.path, "w", encoding="utf-8", buffering=1)
        self._emit = emit
        self._prefix = prefix
        self._count = 0
        self._progress_interval = progress_interval_s
        self._lock = threading.Lock()

    # ---- Reporter --------------------------------------------------------

    def info(self, message: str) -> None:
        self._lines(message)

    def warning(self, message: str) -> None:
        self._lines(message, label="Warning: ")

    def error(self, message: str) -> None:
        self._lines(message, label="Error: ")

    def progress(
        self, items: Iterable[T], desc: str, total: int | None = None
    ) -> Iterator[T]:
        if total is None and hasattr(items, "__len__"):
            total = len(items)  # type: ignore[arg-type]
        started = last = time.monotonic()
        done = 0
        self._emit({"type": "progress", "text": _progress_text(desc, 0, total, 0)})
        try:
            for item in items:
                yield item
                done += 1
                now = time.monotonic()
                if now - last >= self._progress_interval:
                    last = now
                    text = _progress_text(desc, done, total, now - started)
                    self._emit({"type": "progress", "text": text})
        finally:
            self._emit({"type": "progress", "text": ""})
            self.info(_progress_text(desc, done, total, time.monotonic() - started))

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._file.close()

    # ---- internals -------------------------------------------------------

    def _lines(self, message: str, label: str = "") -> None:
        with self._lock:
            for i, text in enumerate(message.rstrip().splitlines() or [""]):
                text = (label if i == 0 else "") + text.rstrip()
                if not text:
                    continue
                line = f"{time.strftime('%H:%M:%S')}  {text}"
                if not self._file.closed:
                    self._file.write(line + "\n")
                self._count += 1
                self._emit({"type": "log", "n": self._count, "line": line})
                print(f"{self._prefix}{text}", flush=True)


def _progress_text(desc: str, done: int, total: int | None, elapsed: float) -> str:
    if total:
        pct = 100 * done / total
        text = f"{desc}: {done}/{total} ({pct:.0f}%)"
        if 0 < done < total:
            eta = elapsed / done * (total - done)
            text += f" · about {_duration(eta)} left"
        return text
    return f"{desc}: {done}"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"
