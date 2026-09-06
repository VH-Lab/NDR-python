"""Assemble one SmartSPIM tile as a 3D volume (z, y, x).

Port of ``+ndr/+format/+smartspim/readTileVolume.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import validate_call

from ndr.format.smartspim.readTileInfo import readTileInfo

__all__ = ["readTileVolume"]

_NUMPY_DTYPE = {"single": "float32", "double": "float64"}


def _tifffile():
    try:
        import tifffile
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "tifffile is required for reading SmartSPIM tiles. "
            'Install with: pip install "ndr[formats]"'
        ) from err
    return tifffile


def _numpy_dtype(matlab_class: str) -> np.dtype:
    return np.dtype(_NUMPY_DTYPE.get(matlab_class, matlab_class))


@validate_call(config={"arbitrary_types_allowed": True})
def readTileVolume(
    rootDir: str,
    channelName: str,
    tileId: str,
    *,
    ZRange: Any = None,
) -> np.ndarray:
    """Read the z-slices of one SmartSPIM tile and stack as a 3D array (z, y, x).

    By default all slices are read. Pass ``ZRange`` as a two-element
    1-based inclusive ``[start, stop]`` to read a sub-range.

    Per-tile only: stitching / cross-tile assembly is out of scope for
    the NDR format layer.

    Errors if the tile directory is missing, ZRange is invalid, or any
    slice fails to decode.
    """
    info = readTileInfo(rootDir, channelName, tileId)
    n = int(info["numSlices"])
    h = int(info["height"])
    w = int(info["width"])

    if (
        ZRange is None
        or (hasattr(ZRange, "size") and ZRange.size == 0)
        or (isinstance(ZRange, (list, tuple)) and len(ZRange) == 0)
    ):
        first_z = 1
        last_z = n
    else:
        arr = np.asarray(ZRange).ravel()
        if arr.size != 2 or not np.all(arr == np.floor(arr)):
            raise ValueError("ZRange must be a 2-element integer vector [start, stop].")
        first_z = int(arr[0])
        last_z = int(arr[1])
        if first_z < 1 or last_z > n or first_z > last_z:
            raise ValueError(
                f"ZRange [{first_z} {last_z}] is not a valid 1-based "
                f"inclusive range in [1, {n}]."
            )

    n_slices = last_z - first_z + 1
    slice_paths = info["slicePaths"]

    tifffile = _tifffile()
    dt = _numpy_dtype(info["dtype"])

    first_path = slice_paths[first_z - 1]
    try:
        first_img = tifffile.imread(first_path)
    except Exception as err:
        raise ValueError(f"Failed to decode {first_path}: {err}") from err
    if first_img.shape[0] != h or first_img.shape[1] != w:
        raise ValueError(
            f"Slice {first_path} has shape {first_img.shape[0]}x"
            f"{first_img.shape[1]}; expected {h}x{w}."
        )
    if first_img.dtype != dt:
        # Match MATLAB behavior: first slice locks the class.
        dt = first_img.dtype

    volume = np.zeros((n_slices, h, w), dtype=dt)
    volume[0, :, :] = first_img

    for i in range(1, n_slices):
        path_i = slice_paths[first_z - 1 + i]
        try:
            img = tifffile.imread(path_i)
        except Exception as err:
            raise ValueError(f"Failed to decode {path_i}: {err}") from err
        if img.shape[0] != h or img.shape[1] != w or img.dtype != dt:
            raise ValueError(
                f"Slice {path_i} has shape {img.shape[0]}x{img.shape[1]} "
                f"({img.dtype}); expected {h}x{w} ({dt})."
            )
        volume[i, :, :] = img

    return volume
