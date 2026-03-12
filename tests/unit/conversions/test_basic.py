import pytest
from common.convertions import basic

@pytest.mark.basic_conversion
def test_meters_to_feet():
    """Test meter to feet conversion."""
    assert abs(basic.meters_to_feet(1.0) - 3.28084) < 0.001
    assert abs(basic.meters_to_feet(10.0) - 32.8084) < 0.001

@pytest.mark.basic_conversion
def test_degrees_to_radians():
    """Test degree to radian conversion."""
    import math
    assert abs(basic.degrees_to_radians(180) - math.pi) < 0.001
    assert abs(basic.degrees_to_radians(90) - math.pi/2) < 0.001

@pytest.mark.basic_conversion
def test_haversine_distance():
    """Test haversine distance calculation."""
    # SF to LA (should be ~559 km)
    distance = basic.haversine_distance(
        lon1=-122.4194, lat1=37.7749,  # San Francisco
        lon2=-118.2437, lat2=34.0522,  # Los Angeles
        unit='kilometers'
    )
    assert 550 < distance < 570  # Approximate check

@pytest.mark.basic_conversion
def test_haversine_same_point():
    """Distance between same point should be 0."""
    distance = basic.haversine_distance(
        lon1=-122.4194, lat1=37.7749,
        lon2=-122.4194, lat2=37.7749,
        unit='meters'
    )
    assert distance == 0.0