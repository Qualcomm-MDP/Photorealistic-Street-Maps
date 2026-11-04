"""Keep credentials out of logs and error messages.

Mapillary takes its token as an access_token query parameter, so a failed
request's error (and any traceback that includes it) contains the token.
"""

import os
import re

SECRET_ENV_VARS = ("MAPILLARY_ACCESS_TOKEN",)
REDACTED = "[REDACTED]"

_TOKEN_PARAM = re.compile(r"(access_token=)[^&\s'\"]+", re.IGNORECASE)


def redact(text: str) -> str:
    """text with access_token values and known secret env values masked."""
    text = _TOKEN_PARAM.sub(rf"\g<1>{REDACTED}", text)
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "").strip().strip("\"'")
        if len(value) >= 8:  # don't mask short placeholder values everywhere
            text = text.replace(value, REDACTED)
    return text
