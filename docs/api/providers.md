# Data Providers API

The provider modules handle communication with external geospatial and imagery APIs. All providers live in `src/common/providers/`.

## OSMClient

Fetches building footprints and metadata from OpenStreetMap via the Overpass API.

::: common.providers.osm.OSMClient
    options:
      show_source: true

### Usage

```python
from common.providers import OSMClient, OSM_MAP_FEATURES
from common.constants import BoundingBox

bbox = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)
client = OSMClient()
data = client.fetch(bbox, [OSM_MAP_FEATURES.BUILDING])
# data["elements"] contains building geometries
```

## OSM_MAP_FEATURES

Enum of supported OSM feature types for querying.

::: common.providers.osm.OSM_MAP_FEATURES
    options:
      show_source: true

## MapillaryClient

Fetches street-level images and camera metadata from the Mapillary API.

::: common.providers.mapillary.MapillaryClient
    options:
      show_source: true

### Usage

```python
from common.providers import MapillaryClient
from common.constants import BoundingBox

bbox = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)
client = MapillaryClient()
data = client.fetch(
    bbox,
    fields=["id", "thumb_original_url", "computed_geometry",
            "computed_compass_angle", "camera_parameters",
            "width", "height"],
    limit=1000,
)
```

!!! note "Authentication"
    The Mapillary client requires a `MAPILLARY_ACCESS_TOKEN` environment variable. See [Getting Started](../getting-started.md) for setup instructions.

## OverpassClient

Low-level client for the Overpass API. Used internally by `OSMClient`.

::: common.providers.overpass.OverpassClient
    options:
      show_source: true

## Data Ingest

The top-level ingest function that orchestrates both providers.

::: data_ingest.ingest.ingest_data
    options:
      show_source: true
