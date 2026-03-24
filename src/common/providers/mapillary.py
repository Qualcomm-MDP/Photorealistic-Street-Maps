import os
import warnings
from ..constants import MAPILLARY_URL, BoundingBox
from scipy.spatial.transform import Rotation as R
import requests


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

    def fetch(self, bbox: BoundingBox, fields: list[str], limit=100):
        # TODO: Create a fetch function that takes in some arguments
        # from the user and return the Mapillary data
        # follow https://github.com/Qualcomm-MDP/Blender_Map_Overlay/blob/main/map_extract.py

        params = {
            "access_token": self.get_token(),
            "fields": ",".join(fields),
            "bbox": bbox.to_str(),
            "limit": limit,
        }

        try:
            resp = requests.get(MAPILLARY_URL, params=params)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API Request Error: {e}")

        data = resp.json().get("data", [])

        if not data:
            raise RuntimeError("No images found.")

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
            }

        return metadata

    # TODO: Add more functions that is needed to process mappilary
    # data. Refer to https://mapillary.github.io/mapillary-python-sdk/docs/intro
