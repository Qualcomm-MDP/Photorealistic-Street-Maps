"""Obstruction classes by name and the class-map cache, without loading a model."""

import numpy as np

from segmentation.model import LabelCache
from segmentation.obstruction import obstruction_mask

VISTAS = {17: "Building", 19: "Person", 55: "Car", 64: "Ego Vehicle", 27: "Sky"}
CITYSCAPES = {2: "building", 11: "person", 13: "car", 10: "sky"}


def _labels(ids):
    """A 40 x 40 class map with one 10 px wide band per class id."""
    seg = np.zeros((40, 40), np.uint8)
    for k, i in enumerate(ids):
        seg[:, 10 * k : 10 * k + 10] = i
    return seg


def test_vistas_obstructions_include_the_capture_vehicle():
    mask = obstruction_mask(_labels([17, 19, 55, 64]), VISTAS)
    assert mask[20, 2] == 0  # building, more than 7 px from the person band
    assert (mask[:, 12:40] == 255).all()  # person, car and ego vehicle


def test_cityscapes_names_work_too():
    mask = obstruction_mask(_labels([2, 10, 13, 2]), CITYSCAPES)
    assert mask[20, 2] == 0 and mask[20, 12] == 0  # building, sky
    assert (mask[:, 20:30] == 255).all()  # car
    assert mask[20, 38] == 0


def test_label_cache_scales_maps_and_drops_another_models(tmp_path):
    cache = LabelCache(tmp_path / "labels")
    cache.start("vistas", VISTAS)
    cache.save("123", _labels([17, 55]))
    assert cache.model() == ("vistas", VISTAS)
    big = cache.get("123", (80, 80))
    assert big.shape == (80, 80) and big[0, 0] == 17 and big[0, 30] == 55
    assert cache.get("missing") is None

    cache.start("cityscapes", CITYSCAPES)  # another model's maps would mislabel classes
    assert cache.get("123") is None and cache.model()[0] == "cityscapes"
