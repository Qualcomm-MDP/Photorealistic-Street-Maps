"""How pipeline code reports what it's doing.

Pipeline code calls a Reporter instead of print()/tqdm directly:

    reporter = get_reporter(state)
    reporter.info("Saved mesh")
    for item in reporter.progress(items, desc="Downloading images"):
        ...

The default Reporter prints to the console with tqdm progress bars, so the CLI
behaves as before. The web API passes its own Reporter in the pipeline metadata
("reporter") that saves a log file and streams lines to the browser.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

from tqdm import tqdm

T = TypeVar("T")


class Reporter:
    """Console reporter used by the CLI and whenever no other one is given."""

    def info(self, message: str) -> None:
        # tqdm.write prints without breaking an active progress bar.
        tqdm.write(message)

    def warning(self, message: str) -> None:
        warnings.warn(message, stacklevel=2)

    def error(self, message: str) -> None:
        tqdm.write(message, file=sys.stderr)

    def progress(
        self, items: Iterable[T], desc: str, total: int | None = None
    ) -> Iterator[T]:
        return iter(tqdm(items, desc=desc, total=total))


CONSOLE = Reporter()


def get_reporter(state: Any = None) -> Reporter:
    """The reporter for this pipeline run (from state metadata), or the console."""
    if state is not None:
        reporter = state.get_metadata("reporter", None)
        if reporter is not None:
            return reporter
    return CONSOLE
