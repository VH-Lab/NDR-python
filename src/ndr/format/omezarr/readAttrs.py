"""Parse the .zattrs of an OME-Zarr store.

Port of ``+ndr/+format/+omezarr/readAttrs.m``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import validate_call

__all__ = ["readAttrs"]


@validate_call
def readAttrs(zarrPath: str) -> dict[str, Any]:
    """Read ``<zarrPath>/.zattrs`` and return the parsed contents.

    Errors if the directory does not exist, ``.zattrs`` is absent, or the
    JSON cannot be decoded. This is the low-level metadata primitive;
    ``listPyramids`` builds on it. Callers who need the full NGFF metadata
    (omero, name, etc.) can read it here without walking a normalized form.
    """
    if not zarrPath:
        raise ValueError("zarrPath must be a non-empty string.")

    root = Path(zarrPath)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {zarrPath}")

    zattrs_path = root / ".zattrs"
    if not zattrs_path.is_file():
        raise FileNotFoundError(f"No .zattrs found in {zarrPath}")

    try:
        with open(zattrs_path, encoding="utf-8") as fh:
            attrs = json.load(fh)
    except ValueError as err:
        raise ValueError(f"Failed to parse {zattrs_path} as JSON: {err}") from err

    if not isinstance(attrs, dict):
        raise ValueError(f"{zattrs_path}: top-level JSON is not an object.")

    return attrs
