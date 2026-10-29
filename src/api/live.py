"""Pushes job updates and pipeline log lines to browsers over WebSockets.

Messages are produced on the JobManager's worker thread, while WebSocket clients live on
the server's asyncio event loop. publish() hands each update to the loop with
call_soon_threadsafe, and every connected client gets it through its own queue.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

# A client that falls this far behind just misses updates; it receives a full
# snapshot again when it reconnects.
_MAX_QUEUED = 1000


class JobHub:
    def __init__(self) -> None:
        self._clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUED)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._clients.discard(q)

    def publish(self, message: dict[str, Any]) -> None:
        """Thread-safe: may be called from any thread."""
        with self._lock:
            loop = self._loop
        if loop is None or loop.is_closed():
            return  # nobody has connected yet
        loop.call_soon_threadsafe(self._fan_out, message)

    def _fan_out(self, message: dict[str, Any]) -> None:
        for q in list(self._clients):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass
