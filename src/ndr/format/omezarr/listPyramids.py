"""Enumerate pyramids in an OME-Zarr store.

Port of ``+ndr/+format/+omezarr/listPyramids.m``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import validate_call

from ndr.format.omezarr.readAttrs import readAttrs

__all__ = ["listPyramids"]


def _get_char_field(entry: dict, name: str) -> str:
    v = entry.get(name)
    if v is None or v == "":
        return ""
    return str(v)


def _normalize_axes(entry: dict) -> list[dict[str, str]]:
    axes = entry.get("axes")
    if not axes:
        return []
    out = []
    for a in axes:
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "name": _get_char_field(a, "name"),
                "type": _get_char_field(a, "type"),
                "unit": _get_char_field(a, "unit"),
            }
        )
    return out


def _extract_coordinate_transforms(d: dict) -> tuple[list[float], list[float]]:
    scale: list[float] = []
    translation: list[float] = []
    ct = d.get("coordinateTransformations")
    if not ct:
        return scale, translation
    for c in ct:
        if not isinstance(c, dict) or "type" not in c:
            continue
        t = str(c["type"])
        if t == "scale" and "scale" in c:
            scale = [float(x) for x in c["scale"]]
        elif t == "translation" and "translation" in c:
            translation = [float(x) for x in c["translation"]]
    if not translation and scale:
        translation = [0.0] * len(scale)
    return scale, translation


def _read_zarray_meta(array_dir: Path) -> tuple[list[int], list[int], str]:
    shape: list[int] = []
    chunks: list[int] = []
    dtype = ""
    zarray_path = array_dir / ".zarray"
    if not zarray_path.is_file():
        return shape, chunks, dtype
    try:
        with open(zarray_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return shape, chunks, dtype
    if "shape" in meta:
        shape = [int(x) for x in meta["shape"]]
    if "chunks" in meta:
        chunks = [int(x) for x in meta["chunks"]]
    if "dtype" in meta:
        dtype = str(meta["dtype"])
    return shape, chunks, dtype


def _normalize_levels(entry: dict, zarr_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    datasets = entry.get("datasets")
    if not datasets:
        return out
    for d in datasets:
        if not isinstance(d, dict):
            continue
        path_str = _get_char_field(d, "path")
        scale, translation = _extract_coordinate_transforms(d)
        array_dir = zarr_root / path_str.replace("/", "/")
        shape, chunks, dtype = _read_zarray_meta(array_dir)
        out.append(
            {
                "path": path_str,
                "shape": shape,
                "chunks": chunks,
                "dtype": dtype,
                "scale": scale,
                "translation": translation,
            }
        )
    return out


@validate_call
def listPyramids(zarrPath: str) -> list[dict[str, Any]]:
    """Return one entry per NGFF ``multiscales`` entry in ``<zarrPath>/.zattrs``.

    Each entry has fields:

    - ``name`` (str, from ``multiscales[i].name``, may be empty)
    - ``type`` (str, descriptive only -- for the lab layout this is "box"
      for mean and "max" for max, but the field's history includes wrong
      values (a previous writer labeled the mean pyramid "gaussian"); do
      not switch behavior on it)
    - ``axes`` (list of {name, type, unit} dicts as they appear in NGFF)
    - ``levels`` (list of dicts, one per resolution level, with fields
      ``path``, ``shape``, ``chunks``, ``dtype``, ``scale``, ``translation``)

    Pyramid selection everywhere else in this package is by NAME. This
    function is the source of truth for what names exist.

    The shared-level-0 case (both pyramids' first entry points at the same
    path, typically '0') is represented honestly: both entries'
    ``levels[0]['path']`` is the same string, and ``resolveArrayPath`` will
    resolve them to the same directory. Nothing here special-cases it.
    """
    attrs = readAttrs(zarrPath)
    if "multiscales" not in attrs:
        raise ValueError(f".zattrs in {zarrPath} has no multiscales field.")

    ms_raw = attrs["multiscales"]
    if not isinstance(ms_raw, list):
        raise ValueError(f"multiscales must be a JSON array; got {type(ms_raw).__name__}.")

    zarr_root = Path(zarrPath)
    pyramids: list[dict[str, Any]] = []
    for entry in ms_raw:
        if not isinstance(entry, dict):
            entry = {}
        pyramids.append(
            {
                "name": _get_char_field(entry, "name"),
                "type": _get_char_field(entry, "type"),
                "axes": _normalize_axes(entry),
                "levels": _normalize_levels(entry, zarr_root),
            }
        )
    return pyramids
