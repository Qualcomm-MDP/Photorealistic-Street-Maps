"""Downloaded Mapillary photos on disk, by image key, so each is downloaded once.

The pose alignment downloads full-resolution photos of the captures it reconstructs into
its ROI folder (data/mapillary/originals/); texturing then reads those instead of
downloading the same photos again.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from common.constants import USER_AGENT


def fetch_bytes(url: str, timeout: float = 120, attempts: int = 4) -> bytes | None:
    """The URL's content, retrying with a growing pause; None when it cannot be fetched
    (Mapillary now and then cannot serve an image)."""
    for attempt in range(attempts):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt < attempts - 1:
                time.sleep(2**attempt)
    return None


class ImageStore:
    """<root>/<key>.jpg: the photo exactly as downloaded."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.jpg"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def get(self, key: str, url: str | None, timeout: float = 120) -> bytes | None:
        """The photo's bytes, downloaded from url only when not stored yet; None when it
        cannot be fetched."""
        p = self.path(key)
        if p.exists():
            return p.read_bytes()
        if not url:
            return None
        data = fetch_bytes(url, timeout)
        if data is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(p)  # never a half-written file under the final name
        return data
