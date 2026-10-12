"""Reader class for LifeCanvas SmartSPIM acquisitions.

Port of ``+ndr/+reader/smartspim.m``.

This class exposes a SmartSPIM acquisition through NDR's image (frame)
API. It is a native NDR image reader, sibling of :class:`ndr.reader.omezarr`
and :class:`ndr.reader.tiffstack.ndr_reader_tiffstack`.

Image sources. A processed SmartSPIM acquisition holds two images of the
same brain (ferret-lightsheet ``formats/SmartSPIM.md``):

- ``All_Channels/`` -- SmartSPIM AutoStitch's stitched volume: the
  destriped tiles merged by TeraStitcher with flat-field and left/right
  illumination correction and blended overlaps. This is the
  better-stitched, harmonized data (and what Imaris shows), so the reader
  PREFERS it whenever it is present for every channel.
- ``Ex_*/`` -- the destriped per-tile stacks, not flat-corrected and not
  stitched. They stay readable tile by tile (they are the source of truth
  and keep the destriper's unscaled intensities), and they are the only
  image source of acquisitions without ``All_Channels``.

The reader never stitches tiles itself: stitched data comes only from
SmartSPIM's own ``All_Channels`` output (issue #128).

Epoch layout. The epochstream names what to read:

- ``[ROOTDIR]`` -- stitched volume, every channel on the C axis
- ``[ROOTDIR, CHANNELNAME]`` -- stitched volume of one channel (C = 1)
- ``[ROOTDIR, CHANNELNAME, TILEID]`` -- one destriped tile (C = 1)

Sidecar files may accompany it inside a larger list. Without a usable
``All_Channels`` folder the first two forms fall back to the per-tile
layout and error with a list of what could complete the pinning.

Dimension model:

- T <- Z axis of the volume (each z-slice TIFF is a "frame")
- Y, X <- Y, X axes of the plane (for ``All_Channels`` the whole
  stitched canvas)
- C = number of channels in the epoch; for ``[ROOTDIR]`` they are
  ordered by their ``_ch<k>`` index and named in
  ``resolveepoch(...)["channelNames"]``
- Z = 1

Frames are returned in ``'YXCZT'`` order: size ``[Y X C 1 nFrames]``.
Timing: SmartSPIM acquisitions do not carry per-slice timestamps.
``epochclock`` is ``'no_time'`` and ``frametimes`` returns NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.smartspim import (
    hasAllChannels,
    isSmartSPIM,
    listChannels,
    listTiles,
    readAllChannelsInfo,
    readAllChannelsVolume,
    readTileInfo,
    readTileVolume,
)
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType

_NUMPY_DTYPE = {"single": "float32", "double": "float64"}


class ndr_reader_smartspim(ndr_reader_base):
    """Reader for SmartSPIM stitched volumes (All_Channels) and tiles."""

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Epoch resolution
    # ------------------------------------------------------------------

    def resolveepoch(self, epochstreams: Any) -> dict[str, Any]:
        """Resolve an epoch to its image source.

        Returns a dict with ``rootDir``, ``channelName`` (``''`` for an
        all-channel epoch), ``tileId`` (``''`` for a stitched epoch),
        ``source`` (``'allChannels'`` or ``'tile'``), ``channelNames``
        (names along the C axis), ``numSlices``, ``height``, ``width``,
        ``dtype``, ``tile`` (the :func:`readTileInfo` result for a tile
        epoch, else None) and ``allChannels`` (the
        :func:`readAllChannelsInfo` result for a stitched epoch, else None).

        ``[ROOTDIR]`` and ``[ROOTDIR, CHANNELNAME]`` read the stitched
        ``All_Channels`` volume when it is usable; otherwise they error
        with a specific message listing what could complete the pinning.
        """
        root_dir, channel_name, tile_id = ndr_reader_smartspim.parseEpochstreams(epochstreams)

        if not root_dir:
            raise ValueError("No SmartSPIM root directory found in epochstream.")
        if not Path(root_dir).is_dir():
            raise NotADirectoryError(f"SmartSPIM root does not exist: {root_dir}")

        info: dict[str, Any] = {
            "rootDir": root_dir,
            "channelName": channel_name,
            "tileId": tile_id,
        }

        if not tile_id and hasAllChannels(root_dir):
            stitched = readAllChannelsInfo(root_dir)
            if not channel_name:
                sel = list(range(len(stitched["channels"])))
                counts = [int(c["numSlices"]) for c in stitched["channels"]]
                if any(c != counts[0] for c in counts):
                    raise ValueError(
                        "All_Channels channels have different slice counts "
                        f"({' '.join(map(str, counts))}) in {root_dir}; pin one "
                        "channel with [rootDir, channelName]."
                    )
            else:
                if channel_name not in stitched["channelNames"]:
                    raise ValueError(
                        f'Channel "{channel_name}" not found. Available in {root_dir}: '
                        f"{ndr_reader_smartspim.formatNameList(stitched['channelNames'])}."
                    )
                sel = [stitched["channelNames"].index(channel_name)]
            info.update(
                {
                    "source": "allChannels",
                    "channelNames": [stitched["channelNames"][i] for i in sel],
                    "numSlices": stitched["channels"][sel[0]]["numSlices"],
                    "height": stitched["height"],
                    "width": stitched["width"],
                    "dtype": stitched["dtype"],
                    "tile": None,
                    "allChannels": stitched,
                }
            )
            return info

        if not channel_name:
            channels = listChannels(root_dir)
            names = [c["name"] for c in channels]
            raise ValueError(
                f"Epochstream does not pin a channel and {root_dir} has no usable "
                "All_Channels stitched volume. "
                f"Available channels: {ndr_reader_smartspim.formatNameList(names)}. "
                "Pass [rootDir, channelName, tileId] as the epochstream."
            )

        if not tile_id:
            tiles = listTiles(root_dir, channel_name)
            ids = [t["id"] for t in tiles]
            raise ValueError(
                f'Epochstream does not pin a tile for channel "{channel_name}" and '
                f"{root_dir} has no usable All_Channels stitched volume. "
                f"Available tiles: {ndr_reader_smartspim.formatNameList(ids)}. "
                "Pass [rootDir, channelName, tileId] as the epochstream."
            )

        tile = readTileInfo(root_dir, channel_name, tile_id)
        info.update(
            {
                "source": "tile",
                "channelNames": [channel_name],
                "numSlices": tile["numSlices"],
                "height": tile["height"],
                "width": tile["width"],
                "dtype": tile["dtype"],
                "tile": tile,
                "allChannels": None,
            }
        )
        return info

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: Any, epoch_select: int = 1) -> int:
        """Return the number of frames (z-slices) in this epoch."""
        info = self.resolveepoch(epochstreams)
        return int(info["numSlices"])

    def framesize(self, epochstreams: Any, epoch_select: int = 1) -> list[int]:
        """Return the ``[Y X C Z T]`` extent of the epoch."""
        info = self.resolveepoch(epochstreams)
        return [
            int(info["height"]),
            int(info["width"]),
            len(info["channelNames"]),
            1,
            int(info["numSlices"]),
        ]

    def dimensionorder(self, epochstreams: Any, epoch_select: int = 1) -> str:
        """Return the dimension order of returned frames."""
        return "YXCZT"

    def datatype(self, epochstreams: Any, epoch_select: int = 1) -> str:
        """Return the underlying numeric class of the pixels (MATLAB class name)."""
        info = self.resolveepoch(epochstreams)
        return info["dtype"]

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
        """Read z-slices from the epoch.

        Returns them in ``'YXCZT'`` order: size
        ``[Y X C 1 numel(frameind)]``, where C is the number of channels
        in the epoch (see ``resolveepoch(...)["channelNames"]``).

        Stitched epochs read each run of consecutive frame indices with
        one :func:`readAllChannelsVolume` call per channel. Tile epochs
        read frames one at a time through :func:`readTileVolume`; large
        contiguous reads that want the extra efficiency can call the
        format functions with a ``ZRange`` directly.
        """
        info = self.resolveepoch(epochstreams)
        Y = int(info["height"])
        X = int(info["width"])
        C = len(info["channelNames"])
        n = int(info["numSlices"])

        if frameind is None or len(np.asarray(frameind).ravel()) == 0:
            frameind = list(range(1, n + 1))
        frameind = [int(f) for f in np.asarray(frameind).ravel()]

        if any(f < 1 or f > n for f in frameind):
            raise IndexError(f"frameind out of range [1, {n}].")

        dt = self.datatype(epochstreams, epoch_select)
        np_dtype = np.dtype(_NUMPY_DTYPE.get(dt, dt))
        frames = np.zeros((Y, X, C, 1, len(frameind)), dtype=np_dtype)

        if info["source"] == "allChannels":
            i = 0
            while i < len(frameind):
                j = i
                while j + 1 < len(frameind) and frameind[j + 1] == frameind[j] + 1:
                    j += 1
                for c, name in enumerate(info["channelNames"]):
                    vol = readAllChannelsVolume(
                        info["rootDir"], name, ZRange=[frameind[i], frameind[j]]
                    )
                    # vol shape is (n, Y, X) -> (Y, X, n)
                    frames[:, :, c, 0, i : j + 1] = np.moveaxis(vol, 0, -1).astype(np_dtype)
                i = j + 1
        else:
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

        Matches :class:`ndr.reader.omezarr`: fluorophore channels of a
        stitched all-channel epoch are returned together on the C axis of
        :meth:`readframes` rather than as separate NDR channels.
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
