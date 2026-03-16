from datetime import datetime, timezone
import math
from pint import UnitRegistry

# initialize Unit Registry
ureg = UnitRegistry()


# Distance conversions
def meters_to_feet(meters: float) -> float:
    """Convert meters to feet"""
    quantity = ureg.Quantity(meters, ureg.meter)
    return float(quantity.to(ureg.feet).magnitude)


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters"""
    quantity = ureg.Quantity(feet, ureg.feet)
    return float(quantity.to(ureg.meter).magnitude)


def kilometers_to_miles(km: float) -> float:
    """Convert kilometers to miles"""
    quantity = ureg.Quantity(km, ureg.kilometer)
    return float(quantity.to(ureg.mile).magnitude)


def miles_to_kilometers(miles: float) -> float:
    """Convert miles to kilometers"""
    quantity = ureg.Quantity(miles, ureg.mile)
    return float(quantity.to(ureg.kilometer).magnitude)


def meters_to_kilometers(meters: float) -> float:
    """Convert meters to kilometers"""
    return meters / 1000.0


def kilometers_to_meters(km: float) -> float:
    """Convert kilometers to meters"""
    return km * 1000.0


# Angle conversions
def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians"""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees"""
    return math.degrees(radians)


def normalize_angle_degrees(angle: float) -> float:
    """Normalize angle to [0, 360) degrees"""
    return angle % 360.0


def normalize_angle_radians(angle: float) -> float:
    """Normalize angle to [0, 2pi) radians"""
    return angle % (2 * math.pi)


def compass_to_cartesian(bearing: float) -> float:
    """
    Convert compass bearing (0° = North, clockwise) to
    cartesian angle (0° = East, counter-clockwise).

    Args:
        bearing: Compass bearing in degrees (0-360)

    Returns:
        Cartesian angle in degrees (0-360)
    """
    return normalize_angle_degrees(90.0 - bearing)


def cartesian_to_compass(angle: float) -> float:
    """
    Convert cartesian angle to compass bearing.

    Args:
        angle: Cartesian angle in degrees (0° = East, counter-clockwise)

    Returns:
        Compass bearing in degrees (0° = North, clockwise)
    """
    return normalize_angle_degrees(90.0 - angle)


# Time conversions
def utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC"""
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def unix_timestamp_to_utc(timestamp: float) -> datetime:
    """Convert Unix timestamp to UTC datetime"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def utc_to_unix_timestamp(dt: datetime) -> float:
    """Convert UTC datetime to Unix timestamp"""
    return dt.timestamp()


# Geographic Distance Calculations
def haversine_distance(
    lon1: float, lat1: float, lon2: float, lat2: float, unit: str = "meters"
) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    Args:
        lon1, lat1: First point coordinates (degrees)
        lon2, lat2: Second point coordinates (degrees)
        unit: Output unit ('meters', 'kilometers', 'miles', 'feet')

    Returns:
        Distance in specified unit
    """
    # Earth radius
    R = 6371000  # meters

    # Convert to radians
    phi1 = degrees_to_radians(lat1)
    phi2 = degrees_to_radians(lat2)
    delta_phi = degrees_to_radians(lat2 - lat1)
    delta_lambda = degrees_to_radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_m = R * c

    # Convert to requested unit
    conversions = {
        "meters": lambda x: x,
        "kilometers": meters_to_kilometers,
        "miles": lambda x: kilometers_to_miles(meters_to_kilometers(x)),
        "feet": meters_to_feet,
    }

    if unit not in conversions:
        raise ValueError(
            f"Unknown unit: {unit}. Use 'meters', 'kilometers', 'miles', or 'feet'"
        )

    return conversions[unit](distance_m)


def euclidean_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate 2D Euclidean distance"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def euclidean_distance_3d(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> float:
    """Calculate 3D Euclidean distance"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


__all__ = [
    # Distance conversions
    "meters_to_feet",
    "feet_to_meters",
    "kilometers_to_miles",
    "miles_to_kilometers",
    "meters_to_kilometers",
    "kilometers_to_meters",
    # Angle conversions
    "degrees_to_radians",
    "radians_to_degrees",
    "normalize_angle_degrees",
    "normalize_angle_radians",
    "compass_to_cartesian",
    "cartesian_to_compass",
    # Timezone utilities
    "utc_now",
    "to_utc",
    "unix_timestamp_to_utc",
    "utc_to_unix_timestamp",
    # Distance calculations
    "haversine_distance",
    "euclidean_distance_2d",
    "euclidean_distance_3d",
    # Unit registry
    "ureg",
]
