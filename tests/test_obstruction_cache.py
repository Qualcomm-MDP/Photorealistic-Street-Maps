import pytest

from segmentation import obstruction


class FakeModel:
    def __init__(self, name, fail_on=None):
        self.name = name
        self.fail_on = fail_on

    def eval(self):
        return self

    def to(self, device):
        if str(device) == self.fail_on:
            raise RuntimeError("CUDA out of memory")
        return self


@pytest.fixture
def fake_models(monkeypatch):
    loads = []

    def model_from_pretrained(model_id):
        loads.append(model_id)
        return FakeModel(model_id, fail_on="cuda")

    monkeypatch.setattr(
        obstruction.Mask2FormerImageProcessor,
        "from_pretrained",
        staticmethod(lambda model_id: object()),
    )
    monkeypatch.setattr(
        obstruction.Mask2FormerForUniversalSegmentation,
        "from_pretrained",
        staticmethod(model_from_pretrained),
    )
    for name in ("_seg_proc", "_seg_model", "_seg_key"):
        monkeypatch.setattr(obstruction, name, None)
    return loads


def test_failed_model_switch_does_not_look_cached(fake_models):
    obstruction._load_seg_model("tiny", "cpu")
    assert obstruction._seg_key == ("tiny", "cpu")

    with pytest.raises(RuntimeError, match="out of memory"):
        obstruction._load_seg_model("large", "cuda")
    assert obstruction._seg_key is None

    # The next run reloads instead of reusing the half-switched model.
    obstruction._load_seg_model("tiny", "cpu")
    assert fake_models == ["tiny", "large", "tiny"]
    assert obstruction._seg_model.name == "tiny"


def test_same_model_and_device_is_reused(fake_models):
    obstruction._load_seg_model("tiny", "cpu")
    obstruction._load_seg_model("tiny", "cpu")

    assert fake_models == ["tiny"]
