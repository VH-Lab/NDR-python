"""Non-throwing probe for a LifeCanvas SmartSPIM raw acquisition directory.

Port of ``+ndr/+format/+smartspim/isSmartSPIM.m``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import validate_call

__all__ = ["isSmartSPIM"]

_CHANNEL_RE = re.compile(r"^Ex_\d+_Em_\w+_Ch\d+$")


@validate_call
def isSmartSPIM(rootDir: str) -> bool:
    """Return True if ``rootDir`` looks like a SmartSPIM raw acquisition.

    The probe requires:

    - ``rootDir`` is a folder
    - ``rootDir/metadata.json`` exists and parses as JSON
    - ``rootDir`` contains at least one channel directory whose name
      matches ``"Ex_<digits>_Em_<chars>_Ch<digits>"``

    Any read/parse failure returns FALSE rather than raising, so callers
    can use this to decide whether to attempt the SmartSPIM code path at
    all. It does not validate the whole acquisition -- the discovery
    functions themselves error with specific messages when their input is
    malformed.
    """
    try:
        root = Path(rootDir)
        if not root.is_dir():
            return False
        meta_path = root / "metadata.json"
        if not meta_path.is_file():
            return False
        with open(meta_path, encoding="utf-8") as fh:
            json.load(fh)
    except (OSError, ValueError):
        return False

    for entry in root.iterdir():
        if entry.is_dir() and _CHANNEL_RE.match(entry.name):
            return True
    return False
