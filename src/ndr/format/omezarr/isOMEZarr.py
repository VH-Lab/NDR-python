"""Non-throwing probe for an OME-Zarr (NGFF v0.4) store.

Port of ``+ndr/+format/+omezarr/isOMEZarr.m``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import validate_call

__all__ = ["isOMEZarr"]


@validate_call
def isOMEZarr(zarrPath: str) -> bool:
    """Return True if ``zarrPath`` is a readable OME-Zarr store.

    Cheap non-throwing probe: TRUE if ``zarrPath`` is a directory holding a
    ``.zattrs`` whose parsed contents carry a non-empty ``multiscales``
    field; FALSE otherwise. Any read or JSON-decode failure returns FALSE
    rather than raising, so callers can use this to decide whether to
    attempt the OME-Zarr code path at all.

    This does not validate the whole NGFF spec -- it only confirms enough
    for the discovery functions to make sense.
    """
    try:
        root = Path(zarrPath)
        if not root.is_dir():
            return False
        zattrs_path = root / ".zattrs"
        if not zattrs_path.is_file():
            return False
        with open(zattrs_path, encoding="utf-8") as fh:
            attrs = json.load(fh)
    except (OSError, ValueError):
        return False

    if not isinstance(attrs, dict):
        return False
    ms = attrs.get("multiscales")
    if not ms:
        return False
    return True
