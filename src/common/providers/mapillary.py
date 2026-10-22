import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ..constants import MAPILLARY_URL, BoundingBox
from scipy.spatial.transform import Rotation as R
import requests

# Mapillary's maximum (and default) page size for /images bbox searches.
MAX_IMAGES_PER_REQUEST = 2000
# bbox searches can't be paged, and in dense areas Mapillary silently returns
# an incomplete subset even well below `limit` (in downtown Ann Arbor 100 m
# tiles still missed ~3% of images; 50 m tiles missed none). So query ~50 m
# tiles in parallel, and split any tile that still comes back near the cap.
TILE_DEG = 0.0005
FETCH_WORKERS = 8
SPLIT_FRACTION = 0.5
MAX_SPLIT_DEPTH = 4
REQUEST_RETRIES = 3


class MapillaryClient:
    def __init__(self):
        # NOTE: Ensure you use `os.environ['MAPILLARY_ACCESS_TOKEN']`
        # when you need to send the mapillary token
        pass

    def get_token(self):
        if "MAPILLARY_ACCESS_TOKEN" not in os.environ:
            raise ValueError(
                "'MAPILLARY_ACCESS_TOKEN' not found on environment variable. Ensure in your `.env` you have 'MAPILLARY_ACCESS_TOKEN=...'"
            )

        return os.environ["MAPILLARY_ACCESS_TOKEN"]

    def _fetch_bbox(
        self, bbox: BoundingBox, fields: list[str], limit: int, depth: int = 0
    ) -> dict[str, dict]:
        params = {
            "access_token": self.get_token(),
            "fields": ",".join(fields),
            "bbox": bbox.to_str(),
            "limit": limit,
        }

        for attempt in range(REQUEST_RETRIES):
            try:
                resp = requests.get(MAPILLARY_URL, params=params, timeout=60)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == REQUEST_RETRIES - 1:
                    raise RuntimeError(f"API Request Error: {e}")
                time.sleep(2**attempt)

        data = resp.json().get("data", [])

        near_cap = len(data) >= SPLIT_FRACTION * limit
        if not near_cap or depth >= MAX_SPLIT_DEPTH:
            if near_cap:
                warnings.warn(
                    f"Mapillary returned {len(data)} images (near the {limit} cap) for "
                    f"{bbox.to_str()} even after splitting; some images may be missing."
                )
            return {img["id"]: img for img in data}

        mid_lat = (bbox.min_lat + bbox.max_lat) / 2
        mid_lon = (bbox.min_lon + bbox.max_lon) / 2
        images: dict[str, dict] = {}
        for lat0, lat1 in ((bbox.min_lat, mid_lat), (mid_lat, bbox.max_lat)):
            for lon0, lon1 in ((bbox.min_lon, mid_lon), (mid_lon, bbox.max_lon)):
                quadrant = BoundingBox(lat0, lat1, lon0, lon1)
                images.update(self._fetch_bbox(quadrant, fields, limit, depth + 1))
        return images

    def fetch(
        self,
        bbox: BoundingBox,
        fields: list[str],
        limit=MAX_IMAGES_PER_REQUEST,
        min_quality: float | None = None,
    ):
        """Fetch image metadata for every Mapillary image in bbox.

        If min_quality is set, images whose quality_score is below it are
        skipped (images with no score are kept). quality_score must then be
        in fields.
        """
        n_lat = max(1, math.ceil((bbox.max_lat - bbox.min_lat) / TILE_DEG))
        n_lon = max(1, math.ceil((bbox.max_lon - bbox.min_lon) / TILE_DEG))
        d_lat = (bbox.max_lat - bbox.min_lat) / n_lat
        d_lon = (bbox.max_lon - bbox.min_lon) / n_lon
        tiles = [
            BoundingBox(
                bbox.min_lat + i * d_lat,
                bbox.min_lat + (i + 1) * d_lat,
                bbox.min_lon + j * d_lon,
                bbox.min_lon + (j + 1) * d_lon,
            )
            for i in range(n_lat)
            for j in range(n_lon)
        ]
        images: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            for tile_images in pool.map(
                lambda t: self._fetch_bbox(t, fields, limit), tiles
            ):
                images.update(tile_images)
        data = list(images.values())

        if not data:
            raise RuntimeError("No images found.")

        if min_quality is not None:
            kept = [
                img
                for img in data
                if img.get("quality_score") is None
                or img["quality_score"] >= min_quality
            ]
            print(
                f"Mapillary quality filter (>= {min_quality}): kept {len(kept)} / "
                f"{len(data)} images, skipped {len(data) - len(kept)}"
            )
            data = kept

        metadata = {}

        for img in data:
            img_id = img.get("id")

            yaw = pitch = roll = None
            computed_rotation = img.get("computed_rotation")

            if computed_rotation:
                try:
                    rot = R.from_rotvec(computed_rotation)
                    yaw_deg, pitch_deg, roll_deg = rot.as_euler("ZYX", degrees=True)
                    yaw = round(float(yaw_deg), 2)
                    pitch = round(float(pitch_deg), 2)
                    roll = round(float(roll_deg), 2)
                except Exception as e:
                    warnings.warn(f"Rotation conversion failed for {img_id}: {e}")
                    continue
            else:
                continue

            focal_px = None
            cam_params = img.get("camera_parameters", [])

            if cam_params and len(cam_params) >= 1:
                f_norm = cam_params[0]
                w = img.get("width")
                h = img.get("height")

                if w and h:
                    max_dim = max(w, h)
                    focal_px = round((f_norm * max_dim), 2)

            metadata[img_id] = {
                "latitude": img["geometry"]["coordinates"][1],
                "longitude": img["geometry"]["coordinates"][0],
                "rotvec": list(computed_rotation),  # raw angle-axis (preferred)
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "focal_length_pixels": focal_px,
                "original_width": img.get("width"),
                "original_height": img.get("height"),
                "image_url": img.get("thumb_original_url"),
                "quality_score": img.get("quality_score"),
            }

        return metadata

    # TODO: Add more functions that is needed to process mappilary
    # data. Refer to https://mapillary.github.io/mapillary-python-sdk/docs/intro
