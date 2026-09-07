"""Describe one SmartSPIM tile without loading pixel data (beyond one header).

Port of ``+ndr/+format/+smartspim/readTileInfo.m``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import validate_call

__all__ = ["readTileInfo"]

_TIFF_SUFFIXES = (".tif", ".tiff")

_MATLAB_CLASS = {"float32": "single", "float64": "double"}


def _tifffile():
    """Import tifffile lazily so it is only required when a TIFF is read."""
    try:
        import tifffile
    except ImportError as err:  # pragma: no cover - exercised only without tifffile
        raise ImportError(
            "tifffile is required for reading SmartSPIM tiles. "
            'Install with: pip install "ndr[formats]"'
        ) from err
    return tifffile


def _tile_id_to_dir(root_dir: str, channel_name: str, tile_id: str) -> Path:
    parts = [p for p in tile_id.split("/") if p]
    return Path(root_dir).joinpath(channel_name, *parts)


def _list_tiff_slices(tile_dir: Path) -> list[str]:
    names = [
        e.name for e in tile_dir.iterdir() if e.is_file() and e.suffix.lower() in _TIFF_SUFFIXES
    ]
    if not names:
        return []
    names = sorted(set(names))
    return [str(tile_dir / n) for n in names]


def _tiff_dtype(dtype: np.dtype) -> str:
    """Return the MATLAB numeric class name for a numpy dtype.

    Mirrors ``tiffDtype`` in the MATLAB helper, except that tifffile has
    already turned BitsPerSample + SampleFormat into a numpy dtype, so we
    just relabel float32/float64 as single/double.
    """
    return _MATLAB_CLASS.get(dtype.name, dtype.name)


@validate_call
def readTileInfo(rootDir: str, channelName: str, tileId: str) -> dict[str, Any]:
    """Return a dict describing one SmartSPIM tile.

    Fields:

    - ``id`` (str): echo of ``tileId``
    - ``channelName`` (str): echo of ``channelName``
    - ``tileDir`` (str): absolute path to the tile's TIFF directory
    - ``numSlices`` (float): number of z-slice files in the tile directory
    - ``slicePaths`` (list[str]): sorted absolute paths to each slice
      (lexical sort; the acquisition writes zero-padded filenames so
      lexical order matches numerical order)
    - ``height`` (float): image height (pixels), from the first slice's
      TIFF header
    - ``width`` (float): image width (pixels), from the first slice's
      TIFF header
    - ``dtype`` (str): MATLAB numeric class of the first slice's pixels
      (e.g. ``"uint16"``, ``"single"``)

    Only the first slice is opened -- reading N slices' headers on
    thousands-of-files tiles would defeat the purpose of a cheap probe.
    """
    if not rootDir:
        raise ValueError("rootDir must be a non-empty string.")
    if not channelName:
        raise ValueError("channelName must be a non-empty string.")
    if not tileId:
        raise ValueError("tileId must be a non-empty string.")

    tile_dir = _tile_id_to_dir(rootDir, channelName, tileId)
    if not tile_dir.is_dir():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    slice_paths = _list_tiff_slices(tile_dir)
    if not slice_paths:
        raise FileNotFoundError(f"No TIFF slices found in tile directory: {tile_dir}")

    tifffile = _tifffile()
    try:
        with tifffile.TiffFile(slice_paths[0]) as tf:
            page = tf.pages[0]
            height = int(page.imagelength)
            width = int(page.imagewidth)
            dtype = np.dtype(page.dtype)
    except Exception as err:
        raise ValueError(f"Failed to read TIFF header of {slice_paths[0]}: {err}") from err

    return {
        "id": tileId,
        "channelName": channelName,
        "tileDir": str(tile_dir),
        "numSlices": float(len(slice_paths)),
        "slicePaths": slice_paths,
        "height": float(height),
        "width": float(width),
        "dtype": _tiff_dtype(dtype),
    }
