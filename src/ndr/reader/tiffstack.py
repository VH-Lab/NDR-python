"""Native multipage-TIFF image reader.

Port of +ndr/+reader/tiffstack.m

The frame API design is adapted from ``nansen.stack.ImageStack`` (VervaekeLab,
https://github.com/VervaekeLab/NANSEN). The following methods map onto
``nansen.stack.ImageStack`` methods of similar purpose::

    ndr.reader (NDR)       | nansen.stack.ImageStack (NANSEN)
    -----------------------|---------------------------------
    readframes             | getFrameSet
    framesize              | getFrameSetSize / Num* + ImageHeight/ImageWidth
    numframes              | NumTimepoints / NumPlanes
    dimensionorder         | DataDimensionOrder
    datatype               | DataType
    frametimes             | getFrameTimes

Only the *design* (method names, dimension model) is adapted; no NANSEN source
code is used, so no NANSEN dependency is introduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType

_TIFF_SUFFIXES = (".tif", ".tiff")


def _tifffile():
    """Import tifffile lazily so it is only required when a TIFF is read."""
    try:
        import tifffile
    except ImportError as err:  # pragma: no cover - exercised only without tifffile
        raise ImportError(
            "tifffile is required for reading TIFF stacks. "
            'Install with: pip install "ndr[formats]"'
        ) from err
    return tifffile


class ndr_reader_tiffstack(ndr_reader_base):
    """Reader for multipage TIFF image stacks.

    Port of ndr.reader.tiffstack. An epoch is one or more ``.tif``/``.tiff``
    files, or a directory containing them, or an anchor/marker file whose
    directory contains them.

    This reader implements the frame API (``numframes``, ``framesize``,
    ``dimensionorder``, ``datatype``, ``frametimes``, ``readframes``) rather
    than the regularly-sampled channel API.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Epoch file resolution
    # ------------------------------------------------------------------

    def imagefiles(self, epochstreams: list[str] | str) -> list[str]:
        """Resolve the epoch streams to the ordered list of TIFF files.

        Entries may be TIFF files listed directly, directories (expanded to
        the TIFFs they contain), or non-image anchor/marker files. Anchors are
        only consulted when no TIFF was listed or found in a directory, in
        which case the anchor's own directory is searched.
        """
        if isinstance(epochstreams, (str, Path)):
            epochstreams = [epochstreams]

        files: list[str] = []
        anchors: list[str] = []

        for entry in epochstreams:
            p = Path(entry)
            if p.is_dir():
                files.extend(self.tiffsindir(entry))
            elif p.suffix.lower() in _TIFF_SUFFIXES:
                files.append(str(entry))
            else:
                anchors.append(str(entry))

        if not files:
            for anchor in anchors:
                files.extend(self.tiffsindir(Path(anchor).parent))

        files = sorted(set(files))  # unique also sorts lexically

        if not files:
            raise ValueError("No .tif/.tiff file found in epoch files or directories.")

        return files

    def tiffsindir(self, folder: str | Path) -> list[str]:
        """Return the TIFF files directly inside ``folder``."""
        folder = Path(folder) if str(folder) else Path(".")
        if not folder.is_dir():
            return []
        return [
            str(f) for f in folder.iterdir() if f.is_file() and f.suffix.lower() in _TIFF_SUFFIXES
        ]

    def filenamefromepochfiles(self, filename_array: list[str]) -> str:
        """Return the first TIFF file of the epoch."""
        return self.imagefiles(filename_array)[0]

    def resolveepoch(self, epochstreams: list[str]) -> dict[str, Any]:
        """Resolve an epoch to its file list, page count, and first-page metadata."""
        tifffile = _tifffile()
        files = self.imagefiles(epochstreams)

        with tifffile.TiffFile(files[0]) as tf:
            pagesperfile = len(tf.pages)
            page = tf.pages[0]
            firstinfo = {
                "Height": int(page.imagelength),
                "Width": int(page.imagewidth),
                "SamplesPerPixel": int(page.samplesperpixel or 1),
                "dtype": np.dtype(page.dtype),
            }

        return {
            "files": files,
            "dirpath": str(Path(files[0]).parent),
            "pagesperfile": pagesperfile,
            "nframes": pagesperfile * len(files),
            "firstinfo": firstinfo,
        }

    def framesource(self, info: dict[str, Any], frameidx: int) -> tuple[str, int]:
        """Map a 1-based frame index onto its (filename, 1-based page)."""
        ppf = info["pagesperfile"]
        fileidx = (frameidx - 1) // ppf + 1
        page = (frameidx - 1) % ppf + 1
        return info["files"][fileidx - 1], page

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: list[str], epoch_select: int = 1) -> int:
        """Return the number of frames across all TIFF files of the epoch."""
        return self.resolveepoch(epochstreams)["nframes"]

    def framesize(self, epochstreams: list[str], epoch_select: int = 1) -> list[int]:
        """Return the ``[Y X C Z T]`` extent of the epoch without reading pixels."""
        info = self.resolveepoch(epochstreams)
        fi = info["firstinfo"]
        return [fi["Height"], fi["Width"], fi["SamplesPerPixel"], 1, info["nframes"]]

    def dimensionorder(self, epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the dimension order of returned frames."""
        return "YXCZT"

    def datatype(self, epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the underlying numeric class of the image pixels.

        Returns a numpy dtype name (e.g. ``'uint16'``, ``'float32'``) where
        MATLAB's ``tiffclass`` returns the MATLAB class name; ``'single'`` and
        ``'double'`` map to ``'float32'`` and ``'float64'`` respectively.
        """
        return ndr_reader_tiffstack.tiffclass(self.resolveepoch(epochstreams)["firstinfo"])

    def frametimes(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the time of each requested frame, or NaN when unknown.

        Times come from a sidecar text file (see ``frametimesfilename``); when
        none exists the epoch is clockless and every frame time is NaN.
        """
        if frameind is None:
            frameind = list(range(1, self.numframes(epochstreams, epoch_select) + 1))
        frameind = np.asarray(frameind, dtype=int)

        if not self.hasframetimes(epochstreams):
            return np.full(len(frameind), np.nan)

        all_t = np.loadtxt(self.frametimesfilename(epochstreams)).ravel()
        return all_t[frameind - 1]

    def readframes(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
        *,
        SelectC: list[int] | np.ndarray | None = None,
        SelectZ: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Read the frames indexed by ``frameind`` as a ``[Y X C Z T]`` array."""
        tifffile = _tifffile()
        info = self.resolveepoch(epochstreams)

        if frameind is None or len(np.asarray(frameind)) == 0:
            frameind = list(range(1, info["nframes"] + 1))
        frameind = [int(f) for f in np.asarray(frameind).ravel()]

        Y, X, C, _Z, _T = self.framesize(epochstreams, epoch_select)
        dt = np.dtype(self.datatype(epochstreams, epoch_select))

        frames = np.zeros((Y, X, C, 1, len(frameind)), dtype=dt)

        # Group by file so each TIFF is opened once, however the caller
        # ordered the frame indices.
        by_file: dict[str, list[tuple[int, int]]] = {}
        for i, f in enumerate(frameind):
            fname, page = self.framesource(info, f)
            by_file.setdefault(fname, []).append((i, page))

        for fname, items in by_file.items():
            with tifffile.TiffFile(fname) as tf:
                for i, page in items:
                    im = tf.pages[page - 1].asarray()
                    frames[:, :, :, 0, i] = im.astype(dt).reshape(Y, X, C)

        return ndr_reader_base.selectframeCZ(frames, SelectC, SelectZ)

    def metadata(self, epochstreams: list[str], epoch_select: int = 1) -> dict[str, Any]:
        """Return image-acquisition metadata.

        A plain TIFF stack carries no raster-scan timing, so this is the
        default "unknown" metadata.
        """
        return ndr_reader_base.emptyimagemetadata()

    # ------------------------------------------------------------------
    # Frame times sidecar
    # ------------------------------------------------------------------

    def frametimesfilename(self, epochstreams: list[str]) -> str:
        """Return the path of the sidecar file holding this epoch's frame times.

        A single-file epoch uses ``<name>_frametimes.txt`` beside it; a
        multi-file epoch uses ``frametimes.txt`` in the containing directory.
        """
        info = self.resolveepoch(epochstreams)
        if len(info["files"]) == 1:
            p = Path(info["files"][0])
            return str(p.parent / f"{p.stem}_frametimes.txt")
        return str(Path(info["dirpath"]) / "frametimes.txt")

    def hasframetimes(self, epochstreams: list[str]) -> bool:
        """Return whether this epoch has a frame-times sidecar file."""
        return Path(self.frametimesfilename(epochstreams)).is_file()

    # ------------------------------------------------------------------
    # Clock and channels
    # ------------------------------------------------------------------

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return ``dev_local_time`` when frame times are known, else ``no_time``."""
        if self.hasframetimes(epochstreams):
            return [ClockType("dev_local_time")]
        return [ClockType("no_time")]

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the first and last frame times, or ``[NaN, NaN]`` when clockless."""
        if self.hasframetimes(epochstreams):
            t = self.frametimes(epochstreams, epoch_select)
            return [[float(t[0]), float(t[-1])]]
        return [[float("nan"), float("nan")]]

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List the channels available in the epoch: a single image channel."""
        return [{"name": "image1", "type": "image", "time_channel": None}]

    # ------------------------------------------------------------------
    # Channel API (not applicable to image readers)
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
        """Not applicable: image readers implement the frame API instead."""
        raise NotImplementedError(
            "ndr_reader_tiffstack is an image reader; use readframes() instead of "
            "readchannels_epochsamples()."
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
        """TIFF stacks carry no native event channels."""
        return np.array([]), np.array([])

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def tiffclass(fi: dict[str, Any]) -> str:
        """Return the numpy dtype name for a resolved TIFF page description.

        MATLAB derives this from ``BitsPerSample`` plus ``SampleFormat`` and
        returns a MATLAB class name; tifffile already resolves the page to a
        numpy dtype, so this returns that dtype's name (``'float32'`` and
        ``'float64'`` where MATLAB says ``'single'`` and ``'double'``).
        """
        return np.dtype(fi["dtype"]).name
