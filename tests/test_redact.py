import pytest
import requests

from common.constants import BoundingBox
from common.providers.mapillary import MapillaryClient
from common.redact import REDACTED, redact

SECRET = "MLY|1234567890|abcdefsecret"


def test_redacts_access_token_parameters_and_secret_values(monkeypatch):
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", SECRET)

    text = redact(f"GET /images?access_token=abc123&bbox=1,2 and {SECRET} again")

    assert text == f"GET /images?access_token={REDACTED}&bbox=1,2 and {REDACTED} again"


def test_leaves_text_alone_without_secrets(monkeypatch):
    monkeypatch.delenv("MAPILLARY_ACCESS_TOKEN", raising=False)

    assert redact("Fetched 12 images") == "Fetched 12 images"


def test_mapillary_request_errors_do_not_carry_the_token(monkeypatch):
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", SECRET)

    def fail(url, params=None, **kwargs):
        raise requests.exceptions.HTTPError(
            f"403 Client Error: Forbidden for url: {url}?access_token={params['access_token']}"
        )

    monkeypatch.setattr(requests, "get", fail)
    bbox = BoundingBox(min_lat=42.29, max_lat=42.30, min_lon=-83.72, max_lon=-83.71)

    with pytest.raises(RuntimeError) as info:
        MapillaryClient().fetch(bbox, ["id"])

    assert SECRET not in str(info.value)
    assert f"access_token={REDACTED}" in str(info.value)
    # The original error (which has the URL) isn't chained into the traceback.
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__
