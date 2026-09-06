"""Reader class for LifeCanvas SmartSPIM raw acquisitions (one tile per epoch).

Port of ``+ndr/+reader/smartspim.m``.

This class exposes one SmartSPIM tile through NDR's image (frame) API.
It is a native NDR image reader, sibling of :class:`ndr.reader.omezarr`
and :class:`ndr.reader.tiffstack.ndr_reader_tiffstack`.

Epoch layout: a SmartSPIM epoch is ONE (channel, tile) pair inside a
SmartSPIM acquisition directory. The epochstream pins all three:
``[ROOTDIR, CHANNELNAME, TILEID]``. Sidecar files may accompany it
inside a larger list. Anything less specific errors with a list of what
could complete the pinning.

Stitching / cross-tile assembly is deliberately out of scope: this
reader returns ONE tile per epoch and reports it as-is. See issue #128.

Dimension model:

- T <- Z axis of the tile (each z-slice TIFF is a "frame")
- Y, X <- Y, X axes of the plane
- C = 1 (one channel per epoch; multi-channel data pins a different
  epoch per channel)
- Z = 1

Frames are returned in ``'YXCZT'`` order: size ``[Y X 1 1 nFrames]``.
Timing: SmartSPIM raw acquisitions do not carry per-slice timestamps.
``epochclock`` is ``'no_time'`` and ``frametimes`` returns NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.smartspim import (
    isSmartSPIM,
    listChannels,
    listTiles,
    readTileInfo,
    readTileVolume,
)
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType

_NUMPY_DTYPE = {"single": "float32", "double": "float64"}


class ndr_reader_smartspim(ndr_reader_base):
    """Reader for one SmartSPIM tile per epoch."""

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Epoch resolution
    # ------------------------------------------------------------------

    def resolveepoch(self, epochstreams: Any) -> dict[str, Any]:
        """Resolve an epoch to ``(rootDir, channelName, tileId)`` plus tile info.

        Returns a dict with ``rootDir``, ``channelName``, ``tileId`` and
        ``tile`` (the :func:`readTileInfo` result for the pinned tile).
        Errors with a specific message listing what could complete the
        pinning when a channel or tile is not pinned.
        """
        root_dir, channel_name, tile_id = ndr_reader_smartspim.parseEpochstreams(epochstreams)

        if not root_dir:
            raise ValueError("No SmartSPIM root directory found in epochstream.")
        if not Path(root_dir).is_dir():
            raise NotADirectoryError(f"SmartSPIM root does not exist: {root_dir}")

        if not channel_name:
            channels = listChannels(root_dir)
            names = [c["name"] for c in channels]
            raise ValueError(
                "Epochstream does not pin a channel. "
                f"Available in {root_dir}: {ndr_reader_smartspim.formatNameList(names)}. "
                "Pass [rootDir, channelName, tileId] as the epochstream."
            )

        if not tile_id:
            tiles = listTiles(root_dir, channel_name)
            ids = [t["id"] for t in tiles]
            raise ValueError(
                f'Epochstream does not pin a tile for channel "{channel_name}". '
                f"Available in {root_dir}: {ndr_reader_smartspim.formatNameList(ids)}. "
                "Pass [rootDir, channelName, tileId] as the epochstream."
            )

        return {
            "rootDir": root_dir,
            "channelName": channel_name,
            "tileId": tile_id,
            "tile": readTileInfo(root_dir, channel_name, tile_id),
        }

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: Any, epoch_select: int = 1) -> int:
        """Return the number of frames (z-slices) in this tile."""
        info = self.resolveepoch(epochstreams)
        return int(info["tile"]["numSlices"])

    def framesize(self, epochstreams: Any, epoch_select: int = 1) -> list[int]:
        """Return the ``[Y X C Z T]`` extent of the tile."""
        info = self.resolveepoch(epochstreams)
        tile = info["tile"]
        return [int(tile["height"]), int(tile["width"]), 1, 1, int(tile["numSlices"])]

    def dimensionorder(self, epochstreams: Any, epoch_select: int = 1) -> str:
        """Return the dimension order of returned frames."""
        return "YXCZT"

    def datatype(self, epochstreams: Any, epoch_select: int = 1) -> str:
        """Return the underlying numeric class of the tile (MATLAB class name)."""
        info = self.resolveepoch(epochstreams)
        return info["tile"]["dtype"]

    def frametimes(
        self,
        epochstreams: Any,
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return NaN for every requested frame (no per-slice timestamps)."""
        if frameind is None:
            frameind = list(range(1, self.numframes(epochstreams, epoch_select) + 1))
        frameind = np.asarray(frameind, dtype=int).ravel()
        return np.full(len(frameind), np.nan)

    def readframes(
        self,
        epochstreams: Any,
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
        *,
        SelectC: list[int] | np.ndarray | None = None,
        SelectZ: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Read z-slices from the pinned tile.

        Reads the z-slices indexed by ``frameind`` from the pinned tile
        and returns them in ``'YXCZT'`` order: size
        ``[Y X 1 1 numel(frameind)]``.

        Frames are read one at a time (the format layer's
        :func:`readTileVolume` can read a contiguous range; for arbitrary
        index lists this is a simple loop). Sparse random reads are fine;
        large contiguous reads that want the extra efficiency can call
        :func:`readTileVolume` with a ``ZRange`` directly.
        """
        info = self.resolveepoch(epochstreams)
        tile = info["tile"]
        Y = int(tile["height"])
        X = int(tile["width"])
        n = int(tile["numSlices"])

        if frameind is None or len(np.asarray(frameind).ravel()) == 0:
            frameind = list(range(1, n + 1))
        frameind = [int(f) for f in np.asarray(frameind).ravel()]

        if any(f < 1 or f > n for f in frameind):
            raise IndexError(f"frameind out of range [1, {n}].")

        dt = self.datatype(epochstreams, epoch_select)
        np_dtype = np.dtype(_NUMPY_DTYPE.get(dt, dt))
        frames = np.zeros((Y, X, 1, 1, len(frameind)), dtype=np_dtype)
        for i, z in enumerate(frameind):
            vol = readTileVolume(
                info["rootDir"],
                info["channelName"],
                info["tileId"],
                ZRange=[z, z],
            )
            # vol shape is (1, Y, X)
            frames[:, :, 0, 0, i] = vol[0, :, :].astype(np_dtype)
        return ndr_reader_base.selectframeCZ(frames, SelectC, SelectZ)

    def epochclock(self, epochstreams: Any, epoch_select: int = 1) -> list[ClockType]:
        """Return ``no_time``: no per-slice timestamps."""
        return [ClockType("no_time")]

    def t0_t1(self, epochstreams: Any, epoch_select: int = 1) -> list[list[float]]:
        """Return ``[[NaN, NaN]]`` for the clockless case."""
        return [[float("nan"), float("nan")]]

    def getchannelsepoch(self, epochstreams: Any, epoch_select: int = 1) -> list[dict[str, Any]]:
        """List the channels available: a single image channel.

        A SmartSPIM epoch pins a specific fluorophore channel via its
        epochstream, so at the NDR-channel level the tile is
        single-channel.
        """
        return [{"name": "image1", "type": "image", "time_channel": None}]

    # ------------------------------------------------------------------
    # Channel API (not applicable to image readers)
    # ------------------------------------------------------------------

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: Any,
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Not applicable: image readers implement the frame API instead."""
        raise NotImplementedError(
            "ndr_reader_smartspim is an image reader; use readframes() "
            "instead of readchannels_epochsamples()."
        )

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: Any,
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SmartSPIM epochs carry no native event channels."""
        return np.array([]), np.array([])

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parseEpochstreams(epochstreams: Any) -> tuple[str, str, str]:
        """Pull ``rootDir``, ``channelName``, ``tileId`` from an epochstream.

        Accepts:

        - ``[ROOTDIR, CHANNELNAME, TILEID]``
        - ``[[ROOTDIR, CHANNELNAME, TILEID], sidecar1, ...]``
        - ``[ROOTDIR]``
        - ``[ROOTDIR, CHANNELNAME]``

        Missing fields come back ``''`` so the caller can raise a
        specific error naming what needs to be pinned.
        """
        root_dir = ""
        channel_name = ""
        tile_id = ""

        if isinstance(epochstreams, (str, Path)):
            return str(epochstreams), channel_name, tile_id

        if not isinstance(epochstreams, (list, tuple)):
            raise TypeError(
                f"Epochstream must be a path or a list; got {type(epochstreams).__name__}."
            )

        # Nested pinning: [[root, channel, tile], ...sidecars]
        for entry in epochstreams:
            if isinstance(entry, (list, tuple)) and len(entry) == 3:
                return str(entry[0]), str(entry[1]), str(entry[2])

        # Flat forms with 1, 2 or 3 str/Path elements
        if all(isinstance(e, (str, Path)) for e in epochstreams):
            if len(epochstreams) == 1:
                return str(epochstreams[0]), channel_name, tile_id
            if len(epochstreams) == 2:
                return str(epochstreams[0]), str(epochstreams[1]), tile_id
            if len(epochstreams) == 3:
                return str(epochstreams[0]), str(epochstreams[1]), str(epochstreams[2])

        # Fallback: find the first SmartSPIM directory in the list.
        for entry in epochstreams:
            if isinstance(entry, (str, Path)) and Path(entry).is_dir() and isSmartSPIM(str(entry)):
                return str(entry), channel_name, tile_id

        return root_dir, channel_name, tile_id

    @staticmethod
    def formatNameList(names: list[str]) -> str:
        if not names:
            return "(none)"
        parts = [f'"{n}"' if n else '""' for n in names]
        return ", ".join(parts)
