"""NDR global variables and paths.

Port of +ndr/globals.m
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ndr.fun.ndrpath import ndrpath


class NDRGlobals:
    """Container for NDR global path and configuration variables."""

    def __init__(self) -> None:
        ndr_root = ndrpath()
        self.path = {
            "path": str(ndr_root),
            "preferences": str(ndr_root / "preferences"),
            "filecachepath": str(ndr_root / "filecache"),
            "temppath": str(Path(tempfile.gettempdir()) / "NDR"),
            "testpath": str(Path(tempfile.gettempdir()) / "NDR" / "test"),
        }
        self.debug = {
            "verbose": False,
        }


# Module-level singleton
ndr_globals = NDRGlobals()
