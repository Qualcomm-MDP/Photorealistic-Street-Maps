from enum import Enum
from typing import Any, Dict, List, Optional
import time
import requests

from ..constants import OVERPASS_FALLBACK_URLS, OVERPASS_URL, USER_AGENT, BoundingBox

# Status codes Overpass returns when it is overloaded or rate limiting.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# Refer https://wiki.openstreetmap.org/wiki/Map_features
class OSM_MAP_FEATURES(Enum):
    BUILDING = "building"
    NATURE = "nature"


class OSMClient:
    def __init__(
        self,
        timeout: int = 60,
        rounds: int = 3,
        backoff_s: float = 5.0,
        urls: tuple[str, ...] = (OVERPASS_URL, *OVERPASS_FALLBACK_URLS),
    ):
        self.timeout = timeout
        self.rounds = rounds
        self.backoff_s = backoff_s
        self.urls = urls

    def fetch(
        self,
        bbox: BoundingBox,
        features: Optional[List[OSM_MAP_FEATURES]] = None,
    ) -> Dict[str, Any]:

        if features is None:
            features = [OSM_MAP_FEATURES.BUILDING]
        query_parts = []

        for feature in features:
            query_parts.extend(self.build_feature_query(feature, bbox=bbox))

        query = f"""
        [out:json][timeout:25];
        (
            {"".join(query_parts)}
        );
        out body geom;
        """
        return self._post(query)

    def _post(self, query: str) -> Dict[str, Any]:
        """POST to each Overpass endpoint in turn, retrying with backoff.

        The public servers regularly time out or rate limit under load, so a
        single failure shouldn't abort the pipeline.
        """
        errors: List[str] = []
        for round_idx in range(self.rounds):
            if round_idx > 0:
                delay = self.backoff_s * 2 ** (round_idx - 1)
                print(f"  Overpass unavailable, retrying in {delay:.0f}s …")
                time.sleep(delay)

            for url in self.urls:
                try:
                    response = requests.post(
                        url,
                        data={"data": query},
                        headers={"User-Agent": USER_AGENT},
                        timeout=self.timeout,
                    )
                except (requests.ConnectionError, requests.Timeout) as exc:
                    errors.append(f"{url}: {type(exc).__name__}")
                    continue

                if response.status_code in RETRYABLE_STATUS:
                    errors.append(f"{url}: HTTP {response.status_code}")
                    continue
                response.raise_for_status()
                return response.json()

        raise RuntimeError(
            f"All Overpass endpoints failed after {self.rounds} rounds: "
            + "; ".join(errors[-len(self.urls) :])
        )

    def build_feature_query(
        self,
        feature: OSM_MAP_FEATURES,
        bbox: BoundingBox,
    ) -> List[str]:
        """Build Overpass query snippets for a feature."""
        tag_filters = self._get_tag_filters(feature)
        query_parts = []

        for tag in tag_filters:
            south, west, north, east = (
                bbox.min_lat,
                bbox.min_lon,
                bbox.max_lat,
                bbox.max_lon,
            )
            area_str = f"({south},{west},{north},{east})"

            query_parts.append(f'node["{tag}"]{area_str};')
            query_parts.append(f'way["{tag}"]{area_str};')
            query_parts.append(f'relation["{tag}"]{area_str};')

        return query_parts

    def _get_tag_filters(self, feature: OSM_MAP_FEATURES) -> List[str]:
        """Map enum feature to OSM tag keys."""
        if feature == OSM_MAP_FEATURES.BUILDING:
            return ["building"]
        elif feature == OSM_MAP_FEATURES.NATURE:
            return ["natural", "landuse", "leisure"]
        else:
            raise ValueError(f"Unsupported feature: {feature}")

    # TODO: Add more functions that is needed to process OSM data
