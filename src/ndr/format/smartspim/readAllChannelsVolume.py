"""Read one channel of SmartSPIM's stitched ``All_Channels`` volume (z, y, x).

Port of ``+ndr/+format/+smartspim/readAllChannelsVolume.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import validate_call

from ndr.format.smartspim.readAllChannelsInfo import _tifffile, readAllChannelsInfo

__all__ = ["readAllChannelsVolume"]

_NUMPY_DTYPE = {"single": "float32", "double": "float64", "logical": "bool"}


@validate_call(config={"arbitrary_types_allowed": True})
def readAllChannelsVolume(
    rootDir: str,
    channelName: str,
    *,
    ZRange: Any = None,
) -> np.ndarray:
    """Read the stitched z-slices of one channel into a ``(z, y, x)`` array.

    Same layout as :func:`readTileVolume`. By default all slices are
    read; ``ZRange`` is a 1-based inclusive ``[start, stop]``.
    ``channelName`` is the channel directory name (e.g.
    ``"Ex_561_Em_561F_Ch2"``); the mapping to the ``_ch<k>`` files is
    resolved by :func:`readAllChannelsInfo`.

    Real All_Channels slices are LZW-compressed, which tifffile decodes
    through the ``imagecodecs`` package.

    Raises :class:`ValueError` if the channel is unknown, ``ZRange`` is
    invalid, or a slice fails to decode or differs in shape/class, and
    the errors of :func:`readAllChannelsInfo` if ``All_Channels`` is not
    usable.
    """
    if not channelName:
        raise ValueError("channelName must be a non-empty string.")
    info = readAllChannelsInfo(rootDir)
    if channelName not in info["channelNames"]:
        raise ValueError(
            f'Channel "{channelName}" not found in {rootDir}. '
            f"Available: {', '.join(info['channelNames'])}"
        )
    ch = info["channels"][info["channelNames"].index(channelName)]
    slice_paths = ch["slicePaths"]
    n = int(ch["numSlices"])

    if ZRange is None or np.asarray(ZRange).size == 0:
        first_z, last_z = 1, n
    else:
        arr = np.asarray(ZRange).ravel()
        if arr.size != 2 or not np.all(arr == np.floor(arr)):
            raise ValueError("ZRange must be a two-element [START STOP] vector.")
        first_z, last_z = int(arr[0]), int(arr[1])
        if first_z < 1 or last_z > n or first_z > last_z:
            raise ValueError(
                f"ZRange [{first_z} {last_z}] is not a valid 1-based "
                f"inclusive range in [1, {n}]."
            )

    h = int(info["height"])
    w = int(info["width"])
    dt = np.dtype(_NUMPY_DTYPE.get(info["dtype"], info["dtype"]))
    volume = np.zeros((last_z - first_z + 1, h, w), dtype=dt)

    tifffile = _tifffile()
    for i, path_i in enumerate(slice_paths[first_z - 1 : last_z]):
        try:
            img = tifffile.imread(path_i)
        except Exception as err:
            hint = (
                " (LZW-compressed slices need the 'imagecodecs' package: "
                "pip install imagecodecs)"
                if "imagecodecs" in str(err)
                else ""
            )
            raise ValueError(f"Failed to decode {path_i}: {err}{hint}") from err
        if img.shape[:2] != (h, w) or img.ndim != 2 or img.dtype != dt:
            raise ValueError(
                f"Slice {path_i} has shape {'x'.join(map(str, img.shape))} "
                f"({img.dtype}); expected {h}x{w} ({dt})."
            )
        volume[i, :, :] = img
    return volume
