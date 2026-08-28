"""Native legacy Prairie View image reader.

Port of +ndr/+reader/prairieview.m

Reads Prairie View two-photon recordings, where one epoch is a directory of
single-page TIFFs plus a configuration file (legacy ``*_Main.pcf`` or a
PVScan/MM-era ``*.xml``). Channel grouping is parsed from the file names, like
NANSEN's PrairieViewTiffs adapter, but no NANSEN code or dependency is
involved.

Timing vs NANSEN: NANSEN's PrairieViewTiffs reads the XML metadata but assumes
a uniform frame period; this reader uses the real per-frame timestamps
recorded in the config, so ``frametimes`` reflects actual acquisition times.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.prairieview.readconfig import readconfig
from ndr.reader.base import ndr_reader_base
from ndr.reader.tiffstack import _MATLAB_CLASS, _tifffile, ndr_reader_tiffstack
from ndr.time.clocktype import ClockType


class ndr_reader_prairieview(ndr_reader_tiffstack):
    """Reader for legacy Prairie View two-photon recordings.

    Port of ndr.reader.prairieview. Inherits the TIFF machinery from
    ``ndr.reader.tiffstack`` and adds Prairie View's multi-channel file
    grouping, per-frame timestamps, and raster-scan metadata.

    A Prairie View "cycle" is one epoch's worth of frames; files are named
    with ``Cycle<N>`` and ``Ch<N>`` markers plus a trailing frame number.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # File layout
    # ------------------------------------------------------------------

    def framelayout(self, epochstreams: list[str]) -> dict[str, Any]:
        """Group the epoch's TIFFs into a (timepoint x channel) grid.

        Parses ``Cycle<N>`` and ``Ch<N>`` out of each file name, plus the
        trailing number as the frame index. Channels sort ascending into the
        C axis; ``(cycle, frame)`` pairs sort into the T axis.

        This mirrors how NANSEN's PrairieViewTiffs adapter groups channels,
        without using any NANSEN code.
        """
        tifffile = _tifffile()
        files = self.imagefiles(epochstreams)
        n = len(files)

        cyc = [1] * n
        ch = [1] * n
        fr = [0] * n

        for i, f in enumerate(files):
            nm = Path(f).stem
            ct = re.search(r"[Cc]ycle(\d+)", nm)
            if ct:
                cyc[i] = int(ct.group(1))
            ht = re.search(r"[Cc]h(\d+)", nm)
            if ht:
                ch[i] = int(ht.group(1))
            fm = re.findall(r"\d+", nm)
            fr[i] = int(fm[-1]) if fm else i + 1

        channels = sorted(set(ch))  # sorted ascending -> C-axis order
        keys = sorted(set(zip(cyc, fr)))  # sorted by cycle then frame
        nT = len(keys)
        nC = len(channels)

        grid: list[list[str | None]] = [[None] * nC for _ in range(nT)]
        key_index = {k: i for i, k in enumerate(keys)}
        chan_index = {c: i for i, c in enumerate(channels)}

        for i, f in enumerate(files):
            grid[key_index[(cyc[i], fr[i])]][chan_index[ch[i]]] = f

        if any(cell is None for row in grid for cell in row):
            raise ValueError(
                "The recording does not have the same set of frames for every "
                "channel; cannot assemble a uniform multi-channel stack."
            )

        with tifffile.TiffFile(grid[0][0]) as tf:
            page = tf.pages[0]
            Y = int(page.imagelength)
            X = int(page.imagewidth)
            dtype = np.dtype(page.dtype)

        return {
            "files": files,
            "channels": channels,
            "keys": keys,
            "grid": grid,
            "Y": Y,
            "X": X,
            "C": nC,
            "nframes": nT,
            "datatype": _MATLAB_CLASS.get(dtype.name, dtype.name),
        }

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: list[str], epoch_select: int = 1) -> int:
        """Return the number of timepoints in the epoch."""
        return self.framelayout(epochstreams)["nframes"]

    def framesize(self, epochstreams: list[str], epoch_select: int = 1) -> list[int]:
        """Return the ``[Y X C Z T]`` extent of the epoch."""
        L = self.framelayout(epochstreams)
        return [L["Y"], L["X"], L["C"], 1, L["nframes"]]

    def datatype(self, epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the underlying numeric class of the image pixels.

        Reports the MATLAB class name, matching
        ``ndr.reader.tiffstack/datatype``.
        """
        return self.framelayout(epochstreams)["datatype"]

    def readframes(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
        *,
        SelectC: list[int] | np.ndarray | None = None,
        SelectZ: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Read the frames indexed by ``frameind`` as a ``[Y X C Z T]`` array.

        ``SelectC`` is honored at the source: unselected channels' files are
        never opened.
        """
        tifffile = _tifffile()
        L = self.framelayout(epochstreams)

        if frameind is None or len(np.asarray(frameind)) == 0:
            frameind = list(range(1, L["nframes"] + 1))
        frameind = [int(f) for f in np.asarray(frameind).ravel()]

        if SelectC is None or len(np.asarray(SelectC)) == 0:
            cidx = list(range(1, L["C"] + 1))
        else:
            cidx = [int(c) for c in np.asarray(SelectC).ravel()]

        dt = ndr_reader_tiffstack.numpy_dtype(L["datatype"])
        frames = np.zeros((L["Y"], L["X"], len(cidx), 1, len(frameind)), dtype=dt)

        for i, ti in enumerate(frameind):
            for j, c in enumerate(cidx):
                fname = L["grid"][ti - 1][c - 1]
                with tifffile.TiffFile(fname) as tf:
                    im = tf.pages[0].asarray()
                frames[:, :, j, 0, i] = im.astype(dt).reshape(L["Y"], L["X"])

        return ndr_reader_base.selectframeCZ(frames, None, SelectZ)

    # ------------------------------------------------------------------
    # Configuration and timing
    # ------------------------------------------------------------------

    def config(self, epochstreams: list[str]) -> dict[str, Any]:
        """Read the Prairie View config that accompanies the epoch's TIFFs."""
        files = self.imagefiles(epochstreams)
        return readconfig(str(Path(files[0]).parent))

    def hasconfigtimes(self, epochstreams: list[str]) -> bool:
        """Return whether the config supplies usable per-frame timestamps."""
        try:
            v = self.config(epochstreams)
        except Exception:
            return False
        ts = v.get("Image_TimeStamp__us_")
        if ts is None or len(np.atleast_1d(ts)) == 0:
            return False
        return not np.all(np.isnan(np.asarray(ts, dtype=float)))

    def frametimes(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the real recorded acquisition time of each requested frame."""
        if frameind is None:
            frameind = list(range(1, self.numframes(epochstreams, epoch_select) + 1))

        if self.hasconfigtimes(epochstreams):
            v = self.config(epochstreams)
            allt = np.asarray(v["Image_TimeStamp__us_"], dtype=float).ravel() / 1e6
            return allt[np.asarray(frameind, dtype=int) - 1]

        return super().frametimes(epochstreams, epoch_select, frameind)

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return ``dev_local_time`` when the config carries frame times."""
        if self.hasconfigtimes(epochstreams):
            return [ClockType("dev_local_time")]
        return super().epochclock(epochstreams, epoch_select)

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the first and last frame times of the epoch."""
        if self.hasconfigtimes(epochstreams):
            t = self.frametimes(epochstreams, epoch_select)
            return [[float(t[0]), float(t[-1])]]
        return super().t0_t1(epochstreams, epoch_select)

    def metadata(self, epochstreams: list[str], epoch_select: int = 1) -> dict[str, Any]:
        """Return the raster-scan acquisition metadata read from the config.

        Fields that the config does not determine stay at their "unknown"
        defaults. All time fields are in seconds.
        """
        m = ndr_reader_base.emptyimagemetadata()

        try:
            v = self.config(epochstreams)
        except Exception:
            return m

        M = v.get("Main")
        if not isinstance(M, dict):
            return m

        def _num(key: str) -> float | None:
            val = M.get(key)
            return (
                float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None
            )

        lpf = _num("Lines_per_frame")
        if lpf is not None:
            m["lines_per_frame"] = lpf

        ppl = _num("Pixels_per_line")
        if ppl is not None:
            m["pixels_per_line"] = ppl

        fp = _num("Frame_period__us_")
        if fp is not None:
            m["frame_period"] = fp / 1e6

        dwell = _num("Dwell_time__us_")
        if dwell is not None:
            m["dwell_time"] = dwell / 1e6

        slp = _num("ScanLine_period__us_")
        if slp is not None:
            m["line_period"] = slp / 1e6
        elif (
            not np.isnan(m["frame_period"])
            and not np.isnan(m["lines_per_frame"])
            and m["lines_per_frame"] > 0
        ):
            m["line_period"] = m["frame_period"] / m["lines_per_frame"]

        if "Bidirectional" in M and M["Bidirectional"] is not None:
            m["bidirectional"] = bool(M["Bidirectional"])

        m["israster"] = not np.isnan(m["frame_period"]) or not np.isnan(m["line_period"])

        return m
