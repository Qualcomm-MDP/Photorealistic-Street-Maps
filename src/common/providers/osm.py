from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import requests

from constants import OVERPASS_URL, BoundingBox

# Refer https://wiki.openstreetmap.org/wiki/Map_features
class OSM_MAP_FEATURES(Enum):
    BUILDING = "building"
    NATURE = "nature"


class OSMClient:
    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def fetch(self,
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
        out body;
        >;
        out skel qt;
        """
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
        
    
    def build_feature_query(
        self,
        feature: OSM_MAP_FEATURES,
        bbox: BoundingBox,
    ) -> List[str]:
        """Build Overpass query snippets for a feature."""
        tag_filters = self._get_tag_filters(feature)
        query_parts = []
        
        for tag in tag_filters:
            south, west, north, east = bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon
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
    