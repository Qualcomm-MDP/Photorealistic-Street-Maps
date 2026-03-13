from dataclasses import dataclass

# Provider URLs
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MAPILLARY_URL = "https://graph.mapillary.com/images"

@dataclass(frozen=True)
class BoundingBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float