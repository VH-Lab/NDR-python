"""Reader for (legacy) Prairie View two-photon recordings.

Port of +ndr/+reader/prairieview.m

This reader reads two-photon image series acquired with Prairie
Technologies' PrairieView software in the LEGACY layout: a recording
directory containing one TIFF per frame plus a ``*_Main.pcf`` configuration
file (or a Prairie View ``.xml``). The per-frame timestamps are read from the
config's ``[Image TimeStamp (us)]`` section (``.pcf``) or from the per-frame
``<Frame absoluteTime>`` (modern PVScan) / ``<Time>`` (legacy MM-era) entries
(``.xml``), so the epoch is a ``dev_local_time`` movie with true (possibly
irregular) per-frame times.

It is a native NDR reader with no external dependencies beyond ``tifffile``.
In MATLAB this class extends ``ndr.reader.tiffstack`` and inherits the
directory/anchor-file resolution and the TIFF frame API; there is no shared
Python ``tiffstack`` reader, so the necessary image behavior (``imagefiles``,
``tiffclass``) is implemented here directly. It overrides the frame layout to
group channels onto the C axis (see :meth:`framelayout`) and the timing
(:meth:`frametimes` / :meth:`epochclock` / :meth:`t0_t1`) to read the real
per-frame timestamps from the Prairie config.

Multi-channel: each channel is written as its own TIFF named
``..._Cycle<n>_Ch<c>_<frame>...``. A "frame" here is one TIMEPOINT; all of a
timepoint's channels are returned together on the C axis of
:meth:`readframes` (in ascending channel number), and there is one timestamp
per timepoint.

Cycles and epochs: NDR reads a collection of cycles as a SINGLE epoch;
:meth:`framelayout` enumerates the frames across all cycles, ordered
cycle-then-frame. So one Prairie run directory == one NDR/NDI epoch.

See also: :func:`ndr.format.prairieview.readconfig`,
:func:`ndr.format.prairieview.configfilename`.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Sequence

import numpy as np

from ndr.format.prairieview.readconfig import readconfig
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType


class ndr_reader_prairieview(ndr_reader_base):
    """Reader for legacy Prairie View two-photon image recordings.

    Port of ndr.reader.prairieview.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # File resolution (adapted from ndr.reader.tiffstack.imagefiles)
    # ------------------------------------------------------------------

    @staticmethod
    def _tiffsindir(folder: str) -> List[str]:
        """Return the full-path .tif/.tiff files in a folder.

        Port of ndr.reader.tiffstack.tiffsindir.
        """
        if not folder:
            folder = "."
        files: List[str] = []
        for pat in ("*.tif", "*.tiff"):
            for f in glob.glob(os.path.join(folder, pat)):
                if not os.path.isdir(f):
                    files.append(f)
        return files

    def imagefiles(self, epochstreams: Sequence[str] | str) -> List[str]:
        """Return the ordered list of TIFF files for an epoch.

        Port of ndr.reader.tiffstack.imagefiles.

        ``epochstreams`` entries may be, in any combination: a TIFF file (used
        directly); a directory (its ``.tif``/``.tiff`` contents are expanded);
        or an anchor file that is not itself a TIFF (e.g. a Prairie
        ``.xml``/``.pcf`` config) -- the TIFFs are then taken from the anchor's
        parent directory, but ONLY when no TIFFs are supplied directly.

        Returns a list of full-path TIFF file names ordered by name. Raises
        ``FileNotFoundError`` if no TIFF files are found.
        """
        if isinstance(epochstreams, str):
            epochstreams = [epochstreams]

        files: List[str] = []  # TIFFs listed directly or expanded from dirs
        anchors: List[str] = []  # non-image files that could anchor a directory
        for entry in epochstreams:
            if os.path.isdir(entry):
                files.extend(self._tiffsindir(entry))
            else:
                ext = os.path.splitext(entry)[1]
                if ext.lower() in (".tif", ".tiff"):
                    files.append(entry)
                else:
                    anchors.append(entry)
        if not files:
            # no TIFFs supplied directly: resolve any anchor files to
            # the TIFFs in their parent directories
            for a in anchors:
                files.extend(self._tiffsindir(os.path.dirname(a)))
        files = sorted(set(files))  # unique also sorts lexically
        if not files:
            raise FileNotFoundError("No .tif/.tiff file found in epoch files or directories.")
        return files

    # ------------------------------------------------------------------
    # Frame layout (channels grouped onto the C axis)
    # ------------------------------------------------------------------

    @staticmethod
    def _tiffclass(dtype: np.dtype) -> str:
        """Return the numpy dtype name of the TIFF pixel data.

        The MATLAB reader derives the class from imfinfo's BitsPerSample /
        SampleFormat; here ``tifffile`` already yields a concrete numpy dtype,
        so its name (e.g. ``'uint16'``, ``'int16'``, ``'float32'``) is the
        equivalent. Mirrors ndr.reader.tiffstack.tiffclass.
        """
        return np.dtype(dtype).name

    def framelayout(self, epochstreams: Sequence[str] | str) -> Dict[str, Any]:
        """Resolve the epoch's frames, grouping channels onto the C axis.

        Port of ndr.reader.prairieview.framelayout.

        Parses the recording's TIFF file names for their Cycle, Channel (Ch)
        and frame-index tokens and builds a timepoint-by-channel grid, so that
        all channels of a timepoint are returned together on the C axis of
        :meth:`readframes`. A "frame" (the unit of :meth:`numframes` and
        :meth:`frametimes`) is one timepoint.

        Files with no ``Ch`` token are treated as a single channel; files with
        no frame digits are ordered by name (their 1-based position).

        Returns
        -------
        dict
            With keys:
              ``files``     ordered TIFF files
              ``channels``  sorted unique channel numbers (C-axis order)
              ``keys``      nframes x 2 [cycle, frame] of each timepoint (sorted)
              ``grid``      nframes x numChannels list-of-lists of file names
              ``Y`` ``X`` ``C``  frame height, width, number of channels
              ``nframes``   number of timepoints
              ``datatype``  underlying numeric class (numpy dtype name)
        """
        import tifffile

        files = self.imagefiles(epochstreams)
        n = len(files)
        cyc = np.ones(n, dtype=np.int64)
        ch = np.ones(n, dtype=np.int64)
        fr = np.zeros(n, dtype=np.int64)
        for i in range(n):
            nm = os.path.splitext(os.path.basename(files[i]))[0]
            ct = re.search(r"[Cc]ycle(\d+)", nm)
            if ct is not None:
                cyc[i] = int(ct.group(1))
            ht = re.search(r"[Cc]h(\d+)", nm)
            if ht is not None:
                ch[i] = int(ht.group(1))
            fm = re.findall(r"\d+", nm)
            if not fm:
                fr[i] = i + 1  # 1-based file index (matches MATLAB fr(i)=i)
            else:
                fr[i] = int(fm[-1])

        channels = np.unique(ch)  # sorted ascending -> C-axis order
        # unique rows of [cyc fr], sorted by cycle then frame
        pairs = np.stack([cyc, fr], axis=1)
        keys = np.unique(pairs, axis=0)  # np.unique sorts rows lexically
        nT = keys.shape[0]
        nC = channels.size

        grid: List[List[Any]] = [[None for _ in range(nC)] for _ in range(nT)]
        # index maps for fast lookup
        key_index = {(int(keys[t, 0]), int(keys[t, 1])): t for t in range(nT)}
        chan_index = {int(channels[c]): c for c in range(nC)}
        for i in range(n):
            ti = key_index[(int(cyc[i]), int(fr[i]))]
            ci = chan_index[int(ch[i])]
            grid[ti][ci] = files[i]

        if any(grid[t][c] is None for t in range(nT) for c in range(nC)):
            raise ValueError(
                "The recording does not have the same set of frames for every "
                "channel; cannot assemble a uniform multi-channel stack."
            )

        with tifffile.TiffFile(grid[0][0]) as tf:
            page = tf.pages[0]
            Y = int(page.imagelength)
            X = int(page.imagewidth)
            dt = self._tiffclass(page.dtype)

        return {
            "files": files,
            "channels": channels,
            "keys": keys,
            "grid": grid,
            "Y": Y,
            "X": X,
            "C": nC,
            "nframes": nT,
            "datatype": dt,
        }

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: Sequence[str] | str, epoch_select: int = 1) -> int:
        """Number of timepoints (frames) in the recording.

        A frame is one timepoint; multiple channels of a timepoint count once.
        See :meth:`framelayout`. Port of ndr.reader.prairieview.numframes.
        """
        L = self.framelayout(epochstreams)
        return int(L["nframes"])

    def framesize(self, epochstreams: Sequence[str] | str, epoch_select: int = 1) -> List[int]:
        """[Y X C Z T] extent, with C = number of channels.

        Port of ndr.reader.prairieview.framesize.
        """
        L = self.framelayout(epochstreams)
        return [int(L["Y"]), int(L["X"]), int(L["C"]), 1, int(L["nframes"])]

    def dimensionorder(self, epochstreams: Sequence[str] | str, epoch_select: int = 1) -> str:
        """The dimension order of returned frames ('YXCZT').

        Port of ndr.reader.tiffstack.dimensionorder (inherited by prairieview).
        """
        return "YXCZT"

    def datatype(self, epochstreams: Sequence[str] | str, epoch_select: int = 1) -> str:
        """Underlying numeric class of the pixel data (numpy dtype name).

        Port of ndr.reader.prairieview.datatype.
        """
        L = self.framelayout(epochstreams)
        return str(L["datatype"])

    # ------------------------------------------------------------------
    # Reading frames
    # ------------------------------------------------------------------

    def readframes(
        self,
        epochstreams: Sequence[str] | str,
        epoch_select: int = 1,
        frameind: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Read timepoints, with all channels on the C axis.

        Port of ndr.reader.prairieview.readframes.

        Returns an array in 'YXCZT' order, size ``[Y, X, C, 1, len(frameind)]``,
        where the C axis holds the recording's channels (in ascending Ch
        number) for each requested timepoint.

        Parameters
        ----------
        epochstreams : sequence of str or str
            File paths / directory comprising the epoch.
        epoch_select : int
            Which epoch to access (a Prairie run is a single epoch).
        frameind : sequence of int, optional
            1-based timepoint indices to read. Defaults to all timepoints.
        """
        import tifffile

        L = self.framelayout(epochstreams)
        if frameind is None:
            frameind = list(range(1, int(L["nframes"]) + 1))
        frameind = np.atleast_1d(np.asarray(frameind, dtype=np.int64))

        Y = int(L["Y"])
        X = int(L["X"])
        C = int(L["C"])
        dt = np.dtype(L["datatype"])
        grid = L["grid"]

        frames = np.zeros((Y, X, C, 1, frameind.size), dtype=dt)
        for i in range(frameind.size):
            ti = int(frameind[i]) - 1  # 1-based -> 0-based
            for ci in range(C):
                im = tifffile.imread(grid[ti][ci])
                frames[:, :, ci, 0, i] = np.asarray(im, dtype=dt).reshape(Y, X)
        return frames

    # ------------------------------------------------------------------
    # Config access + timing
    # ------------------------------------------------------------------

    def config(self, epochstreams: Sequence[str] | str) -> Dict[str, Any]:
        """Read the Prairie config dict for an epoch.

        Resolves the recording directory from ``epochstreams`` and reads its
        Prairie config via :func:`ndr.format.prairieview.readconfig`. Port of
        ndr.reader.prairieview.config.
        """
        files = self.imagefiles(epochstreams)
        dirpath = os.path.dirname(files[0])
        return readconfig(dirpath)

    def hasconfigtimes(self, epochstreams: Sequence[str] | str) -> bool:
        """Does the Prairie config provide per-frame times?

        Returns ``True`` if a config with an ``Image_TimeStamp__us_`` vector is
        present and not all-NaN (so frame times come from the config); ``False``
        otherwise. Port of ndr.reader.prairieview.hasconfigtimes.
        """
        try:
            v = self.config(epochstreams)
            ts = v.get("Image_TimeStamp__us_", None)
            if ts is None:
                return False
            ts = np.asarray(ts, dtype=float)
            return ts.size > 0 and not np.all(np.isnan(ts))
        except Exception:
            return False

    def frametimes(
        self,
        epochstreams: Sequence[str] | str,
        epoch_select: int = 1,
        frameind: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-frame times (seconds) from the Prairie config.

        Port of ndr.reader.prairieview.frametimes.

        Returns the ``[Image TimeStamp (us)]`` values (converted to seconds)
        for the requested frames. Falls back to NaN-per-frame (the
        ``ndr.reader.tiffstack`` no-sidecar behavior) when the config has no
        timestamps.

        Parameters
        ----------
        frameind : sequence of int, optional
            1-based frame indices. Defaults to all frames.
        """
        if frameind is None:
            frameind = list(range(1, self.numframes(epochstreams, epoch_select) + 1))
        frameind = np.atleast_1d(np.asarray(frameind, dtype=np.int64))

        if self.hasconfigtimes(epochstreams):
            v = self.config(epochstreams)
            allt = np.asarray(v["Image_TimeStamp__us_"], dtype=float).ravel() / 1e6
            return allt[frameind - 1]  # 1-based -> 0-based
        # tiffstack fallback: no sidecar -> NaN for each requested frame
        return np.full(frameind.size, np.nan, dtype=float)

    def epochclock(
        self, epochstreams: Sequence[str] | str, epoch_select: int = 1
    ) -> List[ClockType]:
        """Clock type(s) for the epoch.

        Returns ``[ClockType('dev_local_time')]`` when the config provides
        per-frame times; otherwise ``[ClockType('no_time')]`` (the
        ``ndr.reader.tiffstack`` clockless default). Port of
        ndr.reader.prairieview.epochclock.
        """
        if self.hasconfigtimes(epochstreams):
            return [ClockType("dev_local_time")]
        return [ClockType("no_time")]

    def t0_t1(self, epochstreams: Sequence[str] | str, epoch_select: int = 1) -> List[List[float]]:
        """[t0 t1] begin/end times for the epoch.

        From the config timestamps when present; otherwise ``[[NaN, NaN]]``
        (the ``ndr.reader.tiffstack`` clockless default). Port of
        ndr.reader.prairieview.t0_t1.
        """
        if self.hasconfigtimes(epochstreams):
            t = self.frametimes(epochstreams, epoch_select)
            return [[float(t[0]), float(t[-1])]]
        return [[float("nan"), float("nan")]]

    def getchannelsepoch(
        self, epochstreams: Sequence[str] | str, epoch_select: int = 1
    ) -> List[Dict[str, Any]]:
        """List the channels available for an image epoch.

        Returns a single 'image' channel named 'image1'. Multi-channel frames
        are returned together as the C axis of :meth:`readframes` rather than
        as separate NDR channels. Port of
        ndr.reader.tiffstack.getchannelsepoch.
        """
        return [{"name": "image1", "type": "image", "time_channel": None}]

    # ------------------------------------------------------------------
    # Abstract-method implementations (image reader: not sample/event based)
    # ------------------------------------------------------------------

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | List[int],
        epochstreams: List[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Not supported for the image-based Prairie View reader.

        Prairie View is an image reader; frames are read with
        :meth:`readframes`. This mirrors ``ndr.reader.tiffstack``, which
        implements only the frame API and not the regularly-sampled channel
        API.
        """
        raise NotImplementedError(
            "ndr_reader_prairieview is an image reader; use readframes(), not "
            "readchannels_epochsamples()."
        )

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | List[int],
        epochstreams: List[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not supported for the image-based Prairie View reader.

        Prairie View is an image reader; it exposes no event/marker channels.
        Mirrors ``ndr.reader.tiffstack``, which implements only the frame API.
        """
        raise NotImplementedError(
            "ndr_reader_prairieview is an image reader; it has no event channels."
        )
