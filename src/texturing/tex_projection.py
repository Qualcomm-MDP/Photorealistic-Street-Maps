from common import PipelineState
import json


def export_json(data: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def tex_projection(value, state: PipelineState):
    mapillary_data = state.get_metadata("provider_data", {})["mapillary"]
    export_json(mapillary_data, "test.json")

    state.require_metadata("progress_monitor").next()
    return value
