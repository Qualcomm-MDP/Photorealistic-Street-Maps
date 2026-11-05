"""Building heights from OSM tags, shared by the mesh and the pose alignment."""

import pytest

from common.MeshUtils import building_heights, get_base_and_height
from mesh_builder.extrude import extrude_buildings
from tests.test_terrain import BBOX, _building


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"height": "12"}, (0.0, 12.0, "height")),
        ({"height": "12 m"}, (0.0, 12.0, "height")),
        ({"building:height": "8.5"}, (0.0, 8.5, "height")),
        ({"building:levels": "4"}, (0.0, 14.0, "levels")),
        ({}, (0.0, 10.5, "default")),  # untagged: 3 storeys
        ({"height": "30", "min_height": "20"}, (20.0, 30.0, "height")),
        ({"building:levels": "6", "building:min_level": "2"}, (7.0, 21.0, "levels")),
    ],
)
def test_building_heights(tags, expected):
    assert building_heights(tags) == pytest.approx(expected)


def test_a_part_is_at_least_a_storey_above_its_base():
    assert get_base_and_height({"tags": {"min_height": "10", "height": "10"}}) == (
        10.0,
        12.5,
    )


def _extruded_z_up(tags):
    element = _building()
    element["tags"] = {"building": "yes", **tags}
    mesh = extrude_buildings({"elements": [element]}, BBOX)
    return -mesh.vertices[:, 2]  # z-down -> up


def test_untagged_building_is_three_storeys():
    z = _extruded_z_up({})
    assert z.min() == pytest.approx(0.0) and z.max() == pytest.approx(10.5)


def test_part_with_min_height_starts_above_the_ground():
    z = _extruded_z_up({"height": "30", "min_height": "20"})
    assert z.min() == pytest.approx(20.0) and z.max() == pytest.approx(30.0)
