

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", 
        "basic_conversion: Tests for basic conversion functions (distance, angle, haversine)"
    )
