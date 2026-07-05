"""Native multipage-TIFF image/frame reader.

Port of +ndr/+reader/tiffstack.m

This class reads image-series data from TIFF files using only the built-in
``tifffile`` package, with no external MATLAB/hardware dependencies. It is a
native NDR image reader: it implements ONLY the frame API (numframes,
framesize, dimensionorder, datatype, frametimes, readframes, epochclock,
t0_t1) and is a sibling of the regularly-sampled readers (e.g.
ndr.reader.intan_rhd), not a subclass of them.

Epoch layout (single file OR a directory of files):
  An epoch may be described by any of the following EPOCHSTREAMS:
    - a single multipage TIFF file (each page is a frame);
    - a directory (each TIFF in it contributes its page(s), in name order);
    - an explicit list of TIFF files (each contributes its page(s));
    - a non-TIFF anchor/marker file (e.g. a Prairie ``.xml`` config); the
      TIFFs are then taken from the anchor's parent directory, but ONLY when
      no TIFFs are supplied directly.
  Frames are ordered by file name (lexical; acquisition systems zero-pad
  indices) and then by page within each file. The stack is assumed
  homogeneous: every file has the same height/width/channels/datatype and the
  same number of pages as the first file.

Dimension model (v1): the pages across the ordered files are treated as the
time axis T, with one z-plane (Z=1). Channels (C) come from the TIFF's
samples-per-pixel. Frames are returned in 'YXCZT' order.

Timing: frame times come from an optional sidecar text file (one time in
seconds per frame, in frame order). For a single-file epoch the sidecar is
``<tiffbasename>_frametimes.txt``; for a directory/multi-file epoch it is
``frametimes.txt`` in the epoch directory. If a sidecar is present the epoch
is a timeseries with clock 'dev_local_time'; otherwise it is clockless
('no_time') and FRAMETIMES returns NaN.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import tifffile

from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType

_TIFF_EXTS = (".tif", ".tiff")


class ndr_reader_tiffstack(ndr_reader_base):
    """Reader for multipage / multichannel TIFF image stacks.

    Port of ndr.reader.tiffstack.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # File discovery / epoch resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _tiffsindir(folder: str) -> list[str]:
        """Return the full-path .tif/.tiff files in a folder.

        Port of tiffstack.tiffsindir.
        """
        if not folder:
            folder = "."
        files: list[str] = []
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if os.path.isdir(full):
                continue
            if name.lower().endswith(_TIFF_EXTS):
                files.append(full)
        return files

    def imagefiles(self, epochstreams: list[str] | str) -> list[str]:
        """Return the ordered list of TIFF files for an epoch.

        Port of tiffstack.imagefiles.

        EPOCHSTREAMS entries may be, in any combination:
          - a single TIFF file (used directly);
          - a directory (its .tif/.tiff contents are expanded);
          - an ANCHOR file that is not itself a TIFF -- the TIFFs are then
            taken from the anchor's parent directory, but ONLY when no TIFFs
            are supplied directly.

        Returns a list of full-path TIFF file names ordered (and deduplicated)
        lexically. Raises FileNotFoundError if no TIFF files are found.
        """
        if not isinstance(epochstreams, (list, tuple)):
            epochstreams = [epochstreams]

        files: list[str] = []  # TIFFs listed directly or expanded from dirs
        anchors: list[str] = []  # non-image files that could anchor a dir

        for entry in epochstreams:
            if os.path.isdir(entry):
                files.extend(self._tiffsindir(entry))
            else:
                ext = os.path.splitext(entry)[1]
                if ext.lower() in _TIFF_EXTS:
                    files.append(entry)
                else:
                    anchors.append(entry)

        if not files:
            # no TIFFs supplied directly: resolve any anchor files to the
            # TIFFs in their parent directories
            for anchor in anchors:
                files.extend(self._tiffsindir(os.path.dirname(anchor)))

        # unique also sorts lexically (mirrors MATLAB unique)
        files = sorted(set(files))

        if not files:
            raise FileNotFoundError(
                "No .tif/.tiff file found in epoch files or directories."
            )
        return files

    def resolveepoch(self, epochstreams: list[str] | str) -> dict[str, Any]:
        """Resolve an epoch to an ordered frame layout (no pixel read).

        Port of tiffstack.resolveepoch. Returns a dict with keys:
          files        ordered list of TIFF file names
          dirpath      the directory anchoring the epoch (for the sidecar)
          pagesperfile number of pages in each file (from the first file)
          nframes      total number of frames (pagesperfile * numfiles)
          firstinfo    tifffile.TiffFile pages/shape info for the first file
        """
        files = self.imagefiles(epochstreams)
        with tifffile.TiffFile(files[0]) as tf:
            pages = tf.pages
            pagesperfile = len(pages)
            page0 = pages[0]
            firstinfo = {
                "height": int(page0.imagelength),
                "width": int(page0.imagewidth),
                "samplesperpixel": int(getattr(page0, "samplesperpixel", 1) or 1),
                "bitspersample": _first_scalar(page0.bitspersample),
                "sampleformat": int(getattr(page0, "sampleformat", 1) or 1),
                "dtype": np.dtype(page0.dtype),
            }
        return {
            "files": files,
            "dirpath": os.path.dirname(files[0]),
            "pagesperfile": pagesperfile,
            "nframes": pagesperfile * len(files),
            "firstinfo": firstinfo,
        }

    @staticmethod
    def _framesource(info: dict[str, Any], frameidx: int) -> tuple[str, int]:
        """Map a 1-based global frame index to a (file, 0-based page).

        Port of tiffstack.framesource (returns a 0-based page for Python's
        tifffile page indexing).
        """
        ppf = info["pagesperfile"]
        fileidx = (frameidx - 1) // ppf  # 0-based file index
        page = (frameidx - 1) % ppf  # 0-based page index
        return info["files"][fileidx], page

    @staticmethod
    def _filenamefromepochfiles(filename_array: list[str] | str) -> str:
        """Return the first (name-ordered) TIFF file of the epoch.

        Port of tiffstack.filenamefromepochfiles.
        """
        files = ndr_reader_tiffstack().imagefiles(filename_array)
        return files[0]

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: list[str] | str, epoch_select: int = 1) -> int:
        """Number of frames in the epoch (across all files/pages).

        Port of tiffstack.numframes.
        """
        return self.resolveepoch(epochstreams)["nframes"]

    def framesize(
        self, epochstreams: list[str] | str, epoch_select: int = 1
    ) -> list[int]:
        """The [Y X C Z T] extent of the stack, without reading pixels.

        Port of tiffstack.framesize. Y=height, X=width, C=samples per pixel,
        Z=1 (pages/files are treated as T), T=total frames.
        """
        info = self.resolveepoch(epochstreams)
        fi = info["firstinfo"]
        return [fi["height"], fi["width"], fi["samplesperpixel"], 1, info["nframes"]]

    def dimensionorder(
        self, epochstreams: list[str] | str, epoch_select: int = 1
    ) -> str:
        """The dimension order of returned frames ('YXCZT').

        Port of tiffstack.dimensionorder.
        """
        return "YXCZT"

    def datatype(self, epochstreams: list[str] | str, epoch_select: int = 1) -> str:
        """The underlying numeric class of the TIFF pixel data.

        Port of tiffstack.datatype. Returns a MATLAB-style class string
        (e.g. 'uint16', 'int16', 'single').
        """
        info = self.resolveepoch(epochstreams)
        return self.tiffclass(info["firstinfo"])

    def frametimesfilename(self, epochstreams: list[str] | str) -> str:
        """Return the path to the frame-times sidecar.

        Port of tiffstack.frametimesfilename. For a single-file epoch this is
        ``<tiffbasename>_frametimes.txt`` next to the TIFF; for a
        directory/multi-file epoch it is ``frametimes.txt`` in the epoch
        directory. The file may or may not exist; see hasframetimes.
        """
        info = self.resolveepoch(epochstreams)
        if len(info["files"]) == 1:
            p, base = os.path.split(info["files"][0])
            stem = os.path.splitext(base)[0]
            return os.path.join(p, stem + "_frametimes.txt")
        return os.path.join(info["dirpath"], "frametimes.txt")

    def hasframetimes(self, epochstreams: list[str] | str) -> bool:
        """Does this epoch have an explicit frame-times sidecar?

        Port of tiffstack.hasframetimes.
        """
        return os.path.isfile(self.frametimesfilename(epochstreams))

    def frametimes(
        self,
        epochstreams: list[str] | str,
        epoch_select: int = 1,
        frameind: np.ndarray | list[int] | None = None,
    ) -> np.ndarray:
        """The time of each requested frame, in EPOCHCLOCK units.

        Port of tiffstack.frametimes. If a frame-times sidecar exists (one
        time in seconds per frame, in frame order), returns those times for
        FRAMEIND (clock 'dev_local_time'). Otherwise returns NaN for each
        frame (clock 'no_time').

        FRAMEIND is 1-based (mirroring the MATLAB API). If None, all frames
        are returned.
        """
        if frameind is None:
            frameind = np.arange(1, self.numframes(epochstreams, epoch_select) + 1)
        frameind = np.atleast_1d(np.asarray(frameind))

        if self.hasframetimes(epochstreams):
            all_t = np.loadtxt(self.frametimesfilename(epochstreams)).reshape(-1)
            return all_t[frameind - 1]
        return np.full(frameind.shape, np.nan)

    def readframes(
        self,
        epochstreams: list[str] | str,
        epoch_select: int = 1,
        frameind: np.ndarray | list[int] | None = None,
    ) -> np.ndarray:
        """Read frames (TIFF pages across the ordered files).

        Port of tiffstack.readframes. Reads the frames indexed by FRAMEIND
        (1-based) and returns them as an array in 'YXCZT' order:
        shape [Y, X, C, 1, numel(FRAMEIND)].
        """
        info = self.resolveepoch(epochstreams)
        if frameind is None:
            frameind = np.arange(1, info["nframes"] + 1)
        frameind = np.atleast_1d(np.asarray(frameind))

        sz = self.framesize(epochstreams, epoch_select)
        Y, X, C = sz[0], sz[1], sz[2]
        dt = np.dtype(_matlabclass2numpy(self.datatype(epochstreams, epoch_select)))

        frames = np.zeros((Y, X, C, 1, len(frameind)), dtype=dt)
        for i, fidx in enumerate(frameind):
            fname, page = self._framesource(info, int(fidx))
            im = tifffile.imread(fname, key=page)
            frames[:, :, :, 0, i] = np.asarray(im, dtype=dt).reshape(Y, X, C)
        return frames

    def epochclock(
        self, epochstreams: list[str] | str, epoch_select: int = 1
    ) -> list[ClockType]:
        """Return the clock type(s) for an image epoch.

        Port of tiffstack.epochclock. Returns ['dev_local_time'] when a
        frame-times sidecar is present (a movie), and ['no_time'] otherwise
        (an ordered clockless stack / slide scan).
        """
        if self.hasframetimes(epochstreams):
            return [ClockType("dev_local_time")]
        return [ClockType("no_time")]

    def t0_t1(
        self, epochstreams: list[str] | str, epoch_select: int = 1
    ) -> list[list[float]]:
        """Return the [t0 t1] begin/end times of an image epoch.

        Port of tiffstack.t0_t1. For a movie (frame-times sidecar present)
        returns [[firsttime, lasttime]] in dev_local_time. For a clockless
        stack returns [[NaN, NaN]].
        """
        if self.hasframetimes(epochstreams):
            t = self.frametimes(epochstreams, epoch_select)
            return [[float(t[0]), float(t[-1])]]
        return [[float("nan"), float("nan")]]

    def getchannelsepoch(
        self, epochstreams: list[str] | str, epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List the channels available for an image epoch.

        Port of tiffstack.getchannelsepoch. Returns a single 'image' channel
        named 'image1'. Multi-channel TIFFs are returned together as the C
        axis of readframes rather than as separate NDR channels in v1.
        """
        return [{"name": "image1", "type": "image", "time_channel": None}]

    # ------------------------------------------------------------------
    # Abstract methods from the base (not applicable to an image reader).
    # ------------------------------------------------------------------

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Not supported: tiffstack is a frame reader; use readframes."""
        raise NotImplementedError(
            "ndr.reader.tiffstack is an image/frame reader; use readframes()."
        )

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not supported: tiffstack is a frame reader; use readframes."""
        raise NotImplementedError(
            "ndr.reader.tiffstack is an image/frame reader; use readframes()."
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def tiffclass(fi: dict[str, Any]) -> str:
        """Map a TIFF page info dict to a MATLAB numeric class string.

        Port of tiffstack.tiffclass. Given a firstinfo dict (see
        resolveepoch), returns the underlying numeric class string (e.g.
        'uint16', 'int16', 'single') implied by its bits-per-sample and
        sample-format.
        """
        bits = int(fi["bitspersample"])
        fmt = int(fi.get("sampleformat", 1) or 1)
        # TIFF SampleFormat: 1=unsigned int, 2=signed int, 3=ieee float
        if fmt == 3:
            return "single" if bits <= 32 else "double"
        if fmt == 2:
            return f"int{bits}"
        return f"uint{bits}"


def _first_scalar(value: Any) -> int:
    """Return the first element of a possibly-tuple TIFF tag as an int."""
    if isinstance(value, (tuple, list, np.ndarray)):
        return int(value[0])
    return int(value)


def _matlabclass2numpy(cls: str) -> str:
    """Map a MATLAB numeric class string to a numpy dtype name."""
    if cls == "single":
        return "float32"
    if cls == "double":
        return "float64"
    return cls  # uintN / intN are valid numpy names
