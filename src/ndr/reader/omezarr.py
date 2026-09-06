"""Reader class for OME-Zarr (NGFF v0.4) volumes.

Port of ``+ndr/+reader/omezarr.m``.

This class exposes an OME-Zarr store through NDR's image (frame) API.
It is a native NDR image reader, sibling of ``ndr.reader.tiffstack``:
it implements ONLY the frame API (numframes, framesize, dimensionorder,
datatype, frametimes, readframes, epochclock, t0_t1). No fake sample
rate, no fake t0_t1 for the regularly-sampled abstraction.

Everything below this class -- .zattrs and .zarray parsing, pyramid
enumeration, path resolution, pixel decoding -- lives in
:mod:`ndr.format.omezarr`. The class here only maps NDR calls into
``ndr.format.omezarr`` calls.

Epoch layout: an OME-Zarr epoch is a ``.zarr`` DIRECTORY plus a required
pyramid selector. Epochstreams may be, in any combination:

- a 2-element list ``[ZARRPATH, PYRAMIDNAME]`` that pins the pyramid;
- a plain path to the ``.zarr`` directory (errors with the pyramid list --
  pyramid selection has no default, because "which analytical lens am I
  looking through?" cannot be guessed silently; see issue #127);
- additional sidecar files that the file navigator handed along; they
  are ignored here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.omezarr import isOMEZarr, listPyramids, readArray
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType

_MATLAB_CLASS_MAP = {
    "u1": "uint8",
    "i1": "int8",
    "u2": "uint16",
    "i2": "int16",
    "u4": "uint32",
    "i4": "int32",
    "u8": "uint64",
    "i8": "int64",
    "f4": "single",
    "f8": "double",
}


class ndr_reader_omezarr(ndr_reader_base):
    """Reader for OME-Zarr (NGFF v0.4) volumes.

    Dimension model (v1): the NGFF axes (typically c, z, y, x) are mapped
    onto NDR's frame model as:

    - T <- Z axis (each z-plane is a "frame")
    - Y, X <- Y, X axes of the plane
    - C <- C axis (channel dimension)
    - Z = 1

    Frames are returned in ``'YXCZT'`` order: size ``[Y X C 1 nFrames]``.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Epoch resolution
    # ------------------------------------------------------------------

    def resolveepoch(self, epochstreams: Any) -> dict[str, Any]:
        """Resolve an epoch to a ``.zarr`` path and a pinned pyramid.

        Returns a dict with:

        - ``zarrPath``: absolute path to the ``.zarr`` directory
        - ``pyramidName``: the pinned pyramid's name
        - ``pyramid``: the entry from :func:`listPyramids` matching that name
        - ``axisIndex``: dict with keys c, z, y, x giving the (1-based)
          axis positions from the NGFF axes list; NaN if the file does not
          have that axis
        """
        zarr_path, pyramid_name = ndr_reader_omezarr.parseEpochstreams(epochstreams)

        if not pyramid_name:
            pyramids = listPyramids(zarr_path)
            names = [p["name"] for p in pyramids]
            raise ValueError(
                "Epochstream does not pin a pyramid. "
                f"Available in {zarr_path}: {ndr_reader_omezarr.formatNameList(names)}. "
                "Pass [zarrPath, pyramidName] as the epochstream."
            )

        pyramids = listPyramids(zarr_path)
        matches = [i for i, p in enumerate(pyramids) if p["name"] == pyramid_name]
        if not matches:
            names = [p["name"] for p in pyramids]
            raise ValueError(
                f'Pyramid "{pyramid_name}" is not in {zarr_path}. '
                f"Available: {ndr_reader_omezarr.formatNameList(names)}."
            )
        idx = matches[0]

        return {
            "zarrPath": zarr_path,
            "pyramidName": pyramid_name,
            "pyramid": pyramids[idx],
            "axisIndex": ndr_reader_omezarr.classifyAxes(pyramids[idx]["axes"]),
        }

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: Any, epoch_select: int = 1) -> int:
        """Return the number of frames (Z-planes) at level 1."""
        info = self.resolveepoch(epochstreams)
        shape = info["pyramid"]["levels"][0]["shape"]
        z_idx = info["axisIndex"]["z"]
        if z_idx is None or (isinstance(z_idx, float) and np.isnan(z_idx)):
            return 1
        return int(shape[int(z_idx) - 1])

    def framesize(self, epochstreams: Any, epoch_select: int = 1) -> list[int]:
        """Return the ``[Y X C Z T]`` extent of the stack at level 1."""
        info = self.resolveepoch(epochstreams)
        shape = info["pyramid"]["levels"][0]["shape"]
        ax = info["axisIndex"]
        Y = ndr_reader_omezarr.axisSize(shape, ax["y"], 1)
        X = ndr_reader_omezarr.axisSize(shape, ax["x"], 1)
        C = ndr_reader_omezarr.axisSize(shape, ax["c"], 1)
        T = ndr_reader_omezarr.axisSize(shape, ax["z"], 1)
        return [Y, X, C, 1, T]

    def dimensionorder(self, epochstreams: Any, epoch_select: int = 1) -> str:
        """Return the dimension order of returned frames."""
        return "YXCZT"

    def datatype(self, epochstreams: Any, epoch_select: int = 1) -> str:
        """Return the underlying numeric class at level 1 (MATLAB class name)."""
        info = self.resolveepoch(epochstreams)
        return ndr_reader_omezarr.zarrClass(info["pyramid"]["levels"][0]["dtype"])

    def frametimes(
        self,
        epochstreams: Any,
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return NaN for each requested frame.

        NGFF v0.4 permits a ``t`` axis but the lab layout does not carry
        one; this version returns NaN for every requested frame. Promote
        this to read the coordinate transformation when a real time-lapse
        dataset arrives.
        """
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
        Level: int = 1,
        SelectC: list[int] | np.ndarray | None = None,
        SelectZ: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Read frames (Z-planes) from the pinned pyramid.

        Reads the Z-planes indexed by ``frameind`` at the requested
        pyramid ``Level`` (default 1 = highest resolution) and returns them
        in ``'YXCZT'`` order: size ``[Y X numel(C) 1 numel(frameind)]``.

        Level MAY default; pyramid is pinned by the epochstream and cannot
        be overridden here.
        """
        info = self.resolveepoch(epochstreams)
        pyramid_name = info["pyramidName"]
        ax = info["axisIndex"]

        # Frame dimensions come from the SELECTED level, not level 1 --
        # otherwise a Level>1 read allocates the wrong-sized buffer.
        level_shape = info["pyramid"]["levels"][Level - 1]["shape"]
        Y = ndr_reader_omezarr.axisSize(level_shape, ax["y"], 1)
        X = ndr_reader_omezarr.axisSize(level_shape, ax["x"], 1)
        C = ndr_reader_omezarr.axisSize(level_shape, ax["c"], 1)
        dt = self.datatype(epochstreams, epoch_select)

        z_idx = ax["z"]
        z_at_level = ndr_reader_omezarr.axisSize(level_shape, z_idx, 1)
        if frameind is None or len(np.asarray(frameind).ravel()) == 0:
            frameind = list(range(1, z_at_level + 1))
        frameind = [int(f) for f in np.asarray(frameind).ravel()]

        n_axes = len(level_shape)
        start_vec = [1] * n_axes
        stop_vec = list(level_shape)

        np_dtype = ndr_reader_omezarr.numpy_dtype(dt)
        frames = np.zeros((Y, X, C, 1, len(frameind)), dtype=np_dtype)

        for i, fidx in enumerate(frameind):
            if z_idx is not None and not (isinstance(z_idx, float) and np.isnan(z_idx)):
                zi = int(z_idx) - 1
                start_vec[zi] = fidx
                stop_vec[zi] = fidx
            region = np.array([start_vec, stop_vec], dtype=int)
            plane = readArray(
                info["zarrPath"],
                pyramid_name,
                Level,
                Region=region,
                OutputType=dt,
            )
            frames[:, :, :, 0, i] = ndr_reader_omezarr.arrangePlaneYXC(plane, ax, Y, X, C)

        return ndr_reader_base.selectframeCZ(frames, SelectC, SelectZ)

    def epochclock(self, epochstreams: Any, epoch_select: int = 1) -> list[ClockType]:
        """Return ``no_time``: the current version does not assume a ``t`` axis."""
        return [ClockType("no_time")]

    def t0_t1(self, epochstreams: Any, epoch_select: int = 1) -> list[list[float]]:
        """Return ``[[NaN, NaN]]`` for the clockless case."""
        return [[float("nan"), float("nan")]]

    def getchannelsepoch(self, epochstreams: Any, epoch_select: int = 1) -> list[dict[str, Any]]:
        """List the channels available: a single image channel.

        Multi-channel data is returned together on the C axis of
        :meth:`readframes` rather than as separate NDR channels, matching
        the tiffstack convention.
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
            "ndr_reader_omezarr is an image reader; use readframes() instead of "
            "readchannels_epochsamples()."
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
        """OME-Zarr epochs carry no native event channels."""
        return np.array([]), np.array([])

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parseEpochstreams(epochstreams: Any) -> tuple[str, str]:
        """Pull the ``.zarr`` path and pyramid name from an epochstream.

        Accepts a str path, a list containing a ``[path, pyramid]``
        2-list (optionally alongside sidecars), or a flat 2-list
        ``[path, pyramid]``. Returns ``pyramid_name`` as ``''`` when none
        is pinned so callers can raise a specific error naming available
        pyramids.
        """
        pyramid_name = ""
        zarr_path = ""

        if isinstance(epochstreams, (str, Path)):
            return str(epochstreams), pyramid_name

        if not isinstance(epochstreams, (list, tuple)):
            raise TypeError(
                f"Epochstream must be a path or a list; got {type(epochstreams).__name__}."
            )

        # Nested pinning: [[path, pyramid], sidecars...]
        for entry in epochstreams:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                return str(entry[0]), str(entry[1])

        # Flat [path, pyramidName] form -- only when the second element is
        # unambiguously a pyramid name (not itself a path to an existing
        # file/folder).
        if (
            len(epochstreams) == 2
            and isinstance(epochstreams[0], (str, Path))
            and isinstance(epochstreams[1], (str, Path))
            and Path(epochstreams[0]).is_dir()
            and not Path(epochstreams[1]).is_dir()
            and not Path(epochstreams[1]).is_file()
        ):
            return str(epochstreams[0]), str(epochstreams[1])

        for entry in epochstreams:
            if isinstance(entry, (str, Path)) and Path(entry).is_dir() and isOMEZarr(str(entry)):
                return str(entry), pyramid_name

        if not zarr_path:
            raise ValueError("No OME-Zarr directory found in epochstream.")
        return zarr_path, pyramid_name

    @staticmethod
    def classifyAxes(axes: list[dict[str, str]]) -> dict[str, Any]:
        """Assign c/z/y/x indices from an NGFF axes list.

        Missing axes come back as NaN so callers can adapt (e.g. a 2D
        image with no Z or C).
        """
        axis_index: dict[str, Any] = {
            "c": float("nan"),
            "z": float("nan"),
            "y": float("nan"),
            "x": float("nan"),
        }
        for i, a in enumerate(axes):
            nm = a.get("name", "").lower()
            if nm in axis_index:
                axis_index[nm] = i + 1
        return axis_index

    @staticmethod
    def axisSize(shape: list[int], idx: Any, fallback: int) -> int:
        """Return ``shape[idx-1]`` if ``idx`` is a valid 1-based index, else fallback."""
        if idx is None or (isinstance(idx, float) and np.isnan(idx)):
            return fallback
        i = int(idx)
        if i < 1 or i > len(shape):
            return fallback
        return int(shape[i - 1])

    @staticmethod
    def zarrClass(dtypeStr: str) -> str:
        """Map a Zarr dtype string (e.g. ``'<u2'``) to a MATLAB class name."""
        s = str(dtypeStr)
        if s and s[0] in "<>|=":
            core = s[1:]
        else:
            core = s
        if core in _MATLAB_CLASS_MAP:
            return _MATLAB_CLASS_MAP[core]
        raise ValueError(f"Unsupported Zarr dtype: {dtypeStr}")

    @staticmethod
    def numpy_dtype(matlab_class: str) -> np.dtype:
        """Return the numpy dtype for a MATLAB numeric class name."""
        inverse = {"single": "float32", "double": "float64"}
        return np.dtype(inverse.get(matlab_class, matlab_class))

    @staticmethod
    def formatNameList(names: list[str]) -> str:
        if not names:
            return "(none)"
        parts = [f'"{n}"' if n else '""' for n in names]
        return ", ".join(parts)

    @staticmethod
    def arrangePlaneYXC(
        plane: np.ndarray,
        axisIndex: dict[str, Any],
        Y: int,
        X: int,
        C: int,
    ) -> np.ndarray:
        """Permute a :func:`readArray` output to ``[Y, X, C]``.

        ``readArray`` returns an array shaped like the on-disk axes (e.g.
        c-z-y-x); collapse the singleton Z and permute so Y is axis 0, X
        is axis 1, C is axis 2. Missing axes are treated as size 1.
        """
        n_axes = plane.ndim
        dims = list(range(n_axes))
        want_raw = [axisIndex["y"], axisIndex["x"], axisIndex["c"]]
        want: list[int] = []
        for v in want_raw:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            want.append(int(v) - 1)
        rest = [d for d in dims if d not in want]
        perm = tuple(want + rest)
        plane_perm = np.transpose(plane, perm)
        return plane_perm.reshape(Y, X, C)
