import pytest
import requests

from common.constants import BoundingBox
from common.providers import osm
from common.providers.osm import OSMClient

BBOX = BoundingBox(42.29025, 42.29422, -83.71978, -83.71205)


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_post(monkeypatch):
    """Replace requests.post with a queue of canned responses."""
    calls = []
    responses = []

    def _post(url, **kwargs):
        calls.append(url)
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(osm.requests, "post", _post)
    monkeypatch.setattr(osm.time, "sleep", lambda _: None)
    return calls, responses


def make_client():
    return OSMClient(rounds=2, backoff_s=0, urls=("https://a", "https://b"))


def test_falls_back_to_next_endpoint_on_overload(fake_post):
    calls, responses = fake_post
    responses += [FakeResponse(504), FakeResponse(200, {"elements": [1]})]

    assert make_client().fetch(BBOX) == {"elements": [1]}
    assert calls == ["https://a", "https://b"]


def test_retries_another_round_after_all_endpoints_fail(fake_post):
    calls, responses = fake_post
    responses += [
        requests.Timeout(),
        FakeResponse(429),
        FakeResponse(200, {"elements": []}),
    ]

    assert make_client().fetch(BBOX) == {"elements": []}
    assert calls == ["https://a", "https://b", "https://a"]


def test_raises_after_exhausting_rounds(fake_post):
    _, responses = fake_post
    responses += [FakeResponse(504)] * 4

    with pytest.raises(RuntimeError, match="All Overpass endpoints failed"):
        make_client().fetch(BBOX)


def test_does_not_retry_client_errors(fake_post):
    calls, responses = fake_post
    responses += [FakeResponse(400)]

    with pytest.raises(requests.HTTPError):
        make_client().fetch(BBOX)
    assert calls == ["https://a"]
