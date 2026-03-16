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

    def to_list(self):
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]

    def to_str(self):
        return f"{str(self.min_lon)},{str(self.min_lat)},{str(self.max_lon)},{str(self.max_lat)}"

    @classmethod
    def from_json(cls, data):
        required_keys = ["min_lat", "max_lat", "min_lon", "max_lon"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        try:
            min_lat = float(data["min_lat"])
            max_lat = float(data["max_lat"])
            min_lon = float(data["min_lon"])
            max_lon = float(data["max_lon"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric value in bounding box data: {e}")

        return cls(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

    @classmethod
    def from_list(cls, data):
        if len(data) != 4:
            raise ValueError(f"Incorrect list length. Got {len(data)}. Expected 4")

        return cls(
            min_lon=data[0],
            min_lat=data[1],
            max_lon=data[2],
            max_lat=data[3],
        )
