"""Reader for video files (.mp4/.avi/.mov).

Port of +ndr/+reader/video.m

The ``'video'`` reader type is defined by the FILE FORMAT, not by a library: it
reads the frames of a video container (MPEG-4, AVI, QuickTime) and the time
stamped on each frame, so the two ports read the same files and give the same
times. MATLAB implements it with its built-in ``VideoReader``; this port uses
PyAV (``av``), the FFmpeg binding, which is an optional dependency imported
lazily so it is only required when a video is actually read. PyAV wheels ship
their own FFmpeg, so unlike MATLAB on Linux (which needs GStreamer for
MPEG-4/QuickTime) every container FFmpeg understands is readable everywhere.

It is a native NDR image reader: it implements only the frame API
(``numframes``, ``framesize``, ``dimensionorder``, ``datatype``,
``frametimes``, ``readframes``, ``epochclock``, ``t0_t1``,
``getchannelsepoch``, ``metadata``) and is a sibling of
``ndr.reader.tiffstack``.

Epoch layout: an epoch is ONE video file. ``epochstreams`` may also contain
companion files (e.g. an ``epochprobemap.ndi`` that NDI's file navigator hands
along with the video) or a directory; non-video files are ignored and
directories are expanded to the files they contain. Exactly one video file
must remain, otherwise an error is raised.

Dimension model: each video frame is one timepoint T, with one z-plane (Z=1).
Colour channels are the C axis: C=3 for RGB video, C=1 for greyscale video
(grey pixel formats, and palettised video whose palette is grey, which is
what MATLAB's 'Grayscale AVI' profile writes). Frames are returned in
``'YXCZT'`` order. ``frameind`` is a 1-based index into T.

Timing: a video has real frame times, so the epoch clock is always
``dev_local_time`` and ``frametimes`` are seconds from the start of the file.
A frame's time is its PRESENTATION TIME, the timestamp the container stores for
that frame, measured from the start of the file. It is read, not assumed,
which is correct for variable-frame-rate files, where ``(k-1)/FrameRate`` would
drift. Here it is FFmpeg's presentation timestamp relative to the stream's
start time, recorded for every frame as the file is decoded once in order (in
MATLAB, ``VideoReader.CurrentTime`` before each ``readFrame``). Because that scan decodes every frame, its result is cached per file
(keyed by full path, size and modification time) for the rest of the Python
session, and only ``frametimes`` (and ``numframes``, when the container
reports no count) runs it. ``t0_t1`` needs only the first and last frame
times, so it measures them the same way without the scan (see ``endtimes``).

The frame API design is adapted from ``nansen.stack.ImageStack`` (VervaekeLab,
https://github.com/VervaekeLab/NANSEN), as for ``ndr.reader.tiffstack``; only
the design (method names, dimension model) is adapted, no NANSEN code is used.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType

# Session caches of the per-file measurements, keyed as MATLAB keys its
# persistent containers.Map: full path, byte size and modification time.
_SCAN_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_ENDTIMES_CACHE: dict[str, list[float]] = {}
_PTS_CACHE: dict[str, np.ndarray] = {}


def _av():
    """Import PyAV lazily so it is only required when a video is read."""
    try:
        import av
    except ImportError as err:  # pragma: no cover - exercised only without av
        raise ImportError(
            "PyAV (av) is required for reading video files. "
            'Install with: pip install "ndr[formats]"'
        ) from err
    return av


def _cachekey(filename: str) -> str:
    """Return the cache key of a video file, erroring if it does not exist."""
    p = Path(filename)
    if not p.is_file():
        raise FileNotFoundError(f"Video file {filename} not found.")
    st = p.stat()
    return f"{os.path.abspath(filename)}|{st.st_size}|{st.st_mtime_ns}"


def _videostream(v):
    """Return the first video stream of an open container."""
    if not v.streams.video:
        raise ValueError(f"{v.name} contains no video stream.")
    return v.streams.video[0]


def _origin(stream) -> int:
    """The pts that is time zero: the stream's start time, or 0 if unknown."""
    return stream.start_time if stream.start_time is not None else 0


def _frametime(frame, stream) -> float:
    """Presentation time of a decoded frame, in seconds from the start of the file."""
    return float((frame.pts - _origin(stream)) * stream.time_base)


def _framerate(stream) -> float:
    """The stream's average frame rate (MATLAB's FrameRate), or NaN if unknown."""
    rate = stream.average_rate or stream.guessed_rate
    return float(rate) if rate else float("nan")


def _frame2array(frame) -> np.ndarray:
    """Convert a decoded frame to a ``Y x X x C`` array, as VideoReader returns it.

    Greyscale pixel formats give C=1 (``uint8``, or ``uint16`` above 8 bits,
    VideoReader's 'Grayscale'/'Mono8'/'Mono16'); everything else is converted
    to RGB, C=3 (``uint8``, or ``uint16`` above 8 bits, as 'RGB24'/'RGB48').
    Palettised frames are looked up in their palette, and give C=1 when the
    palette is grey, which is how MATLAB's 'Grayscale AVI' profile is stored.
    """
    name = frame.format.name
    if name == "pal8":
        h, w = frame.height, frame.width
        idxplane = frame.planes[0]
        stride = idxplane.line_size
        idx = np.frombuffer(idxplane, np.uint8).reshape(h, abs(stride))[:, :w]
        if stride < 0:
            # A bottom-up DIB (as MATLAB's VideoWriter writes AVI) decodes with
            # a negative stride; the buffer holds the rows bottom row first.
            idx = idx[::-1]
        # FFmpeg stores the palette as native-endian uint32 ARGB: bytes B,G,R,A.
        pal = np.frombuffer(frame.planes[1], np.uint8)[: 256 * 4].reshape(256, 4)
        rgbpal = pal[:, [2, 1, 0]]
        if np.all(rgbpal == rgbpal[:, :1]):
            return rgbpal[idx, :1]
        return rgbpal[idx]
    bits = max(c.bits for c in frame.format.components)
    if name.startswith(("gray", "mono")):
        target = "gray16le" if bits > 8 else "gray"
    else:
        target = "rgb48le" if bits > 8 else "rgb24"
    a = frame.reformat(format=target).to_ndarray()
    if a.ndim == 2:
        a = a[:, :, np.newaxis]
    return a


class ndr_reader_video(ndr_reader_base):
    """Reader for video files, backed by PyAV (FFmpeg).

    Port of ndr.reader.video. An epoch is exactly one video file, optionally
    listed alongside companion files or inside a directory.

    This reader implements the frame API (``numframes``, ``framesize``,
    ``dimensionorder``, ``datatype``, ``frametimes``, ``readframes``) rather
    than the regularly-sampled channel API.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Epoch file resolution
    # ------------------------------------------------------------------

    def videofile(self, epochstreams: list[str] | str) -> str:
        """Return the single video file of an epoch.

        ``epochstreams`` entries may be files or directories (directories are
        expanded to the files they contain). Files whose extension is not a
        recognized video extension (see ``videoextensions``) are ignored, so
        companion files such as ``epochprobemap.ndi`` do not interfere.
        Raises ``ValueError`` if there is not exactly one video file.
        """
        if isinstance(epochstreams, (str, Path)):
            epochstreams = [epochstreams]

        files: list[str] = []
        for entry in epochstreams:
            p = Path(entry)
            if p.is_dir():
                files.extend(str(f) for f in p.iterdir() if f.is_file())
            else:
                files.append(str(entry))

        exts = ndr_reader_video.videoextensions()
        vidfiles = sorted({f for f in files if Path(f).suffix.lower() in exts})

        if not vidfiles:
            raise ValueError(
                "No video file found in the epoch files. Recognized extensions: "
                + ", ".join(exts)
                + "."
            )
        if len(vidfiles) > 1:
            raise ValueError(
                "An ndr.reader.video epoch must contain exactly one video file, "
                f"but {len(vidfiles)} were found: {', '.join(vidfiles)}"
            )
        return vidfiles[0]

    def videoobject(self, epochstreams: list[str] | str):
        """Open the epoch's video file and return the PyAV input container.

        The counterpart of MATLAB's ``VideoReader`` object. The caller owns
        the container and should close it (it is a context manager).
        """
        return _av().open(self.videofile(epochstreams))

    # ------------------------------------------------------------------
    # Frame API
    # ------------------------------------------------------------------

    def numframes(self, epochstreams: list[str] | str, epoch_select: int = 1) -> int:
        """Return the number of frames in the video.

        Uses the container's frame count. For containers that report none
        (e.g. Matroska), the frames are counted by decoding the file once (see
        ``frametimes``; the result is cached).
        """
        with self.videoobject(epochstreams) as v:
            n = ndr_reader_video.videonumframes(v)
            name = v.name
        if math.isnan(n):
            p = Path(name)
            n = len(ndr_reader_video.scanframetimes(str(p.parent), p.name))
        return int(n)

    def framesize(self, epochstreams: list[str] | str, epoch_select: int = 1) -> list[int]:
        """Return the ``[Y X C Z T]`` extent of the video.

        Y = height, X = width, C = colour channels (3 for RGB, 1 for
        greyscale), Z = 1, T = number of frames. C is taken from the shape of
        the first decoded frame, so it always agrees with what ``readframes``
        returns.
        """
        with self.videoobject(epochstreams) as v:
            C, _dt = ndr_reader_video.channelsanddatatype(v)
            cc = _videostream(v).codec_context
            Y, X = int(cc.height), int(cc.width)
        T = self.numframes(epochstreams, epoch_select)
        return [Y, X, C, 1, T]

    def dimensionorder(self, epochstreams: list[str] | str, epoch_select: int = 1) -> str:
        """Return the dimension order of returned frames (``'YXCZT'``)."""
        return "YXCZT"

    def datatype(self, epochstreams: list[str] | str, epoch_select: int = 1) -> str:
        """Return the numeric class of the video frames (usually ``'uint8'``).

        Taken from the class of the first decoded frame. The names are the
        MATLAB class names, which numpy spells identically for the integer
        types a video decodes to.
        """
        with self.videoobject(epochstreams) as v:
            _C, dt = ndr_reader_video.channelsanddatatype(v)
        return dt

    def frametimes(
        self,
        epochstreams: list[str] | str,
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the time of each requested frame, in seconds from the start of the file.

        The times are on the ``dev_local_time`` clock, measured by decoding the
        file once (see the module docstring); the scan is cached per file. If
        ``frameind`` (1-based) is omitted, all frames are returned.
        """
        p = Path(self.videofile(epochstreams))
        all_t = ndr_reader_video.scanframetimes(str(p.parent), p.name)
        if frameind is None or len(np.atleast_1d(frameind)) == 0:
            frameind = np.arange(1, len(all_t) + 1)
        frameind = np.atleast_1d(np.asarray(frameind, dtype=float)).ravel()
        ndr_reader_video.checkframeind(frameind, len(all_t))
        return all_t[frameind.astype(int) - 1]

    def readframes(
        self,
        epochstreams: list[str] | str,
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
        *,
        SelectC: list[int] | np.ndarray | None = None,
        SelectZ: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Read video frames as a ``[Y X C Z T]`` array.

        Reads the frames indexed by ``frameind`` (1-based, any order, random
        access) and returns them with size ``[Y X numel(C) numel(Z)
        numel(frameind)]``. Runs of consecutive indices are read with one seek
        each. The colour channels of a frame are decoded together, so
        ``SelectC``/``SelectZ`` are applied by post-selection.
        """
        filename = self.videofile(epochstreams)
        n = self.numframes(epochstreams, epoch_select)
        if frameind is None or len(np.atleast_1d(frameind)) == 0:
            frameind = np.arange(1, n + 1)
        frameind = np.atleast_1d(np.asarray(frameind, dtype=float)).ravel()
        ndr_reader_video.checkframeind(frameind, n)
        frameind = frameind.astype(int)

        # The pts of each frame, in presentation order, so a frame index can
        # be sought to. The demuxed packet timestamps give it without
        # decoding; if they do not account for every frame, the decoded scan
        # (which defines the frames) does.
        pts = ndr_reader_video._framepts(filename)
        if len(pts) != n:
            pts = ndr_reader_video._scan(filename)[0]
        position = {int(p): k for k, p in enumerate(pts, start=1)}

        with _av().open(filename) as v:
            stream = _videostream(v)
            C, dt = ndr_reader_video.channelsanddatatype(v)
            Y, X = int(stream.codec_context.height), int(stream.codec_context.width)
            frames = np.zeros((Y, X, C, 1, len(frameind)), dtype=dt)

            # split frameind into runs of consecutive increasing indices
            breaks = [0, *(np.flatnonzero(np.diff(frameind) != 1) + 1), len(frameind)]
            for r in range(len(breaks) - 1):
                run = frameind[breaks[r] : breaks[r + 1]]
                first, last = int(run[0]), int(run[-1])
                if first > len(pts) or last > len(pts):
                    raise IndexError(f"Frame {last} could not be located in {filename}.")
                v.seek(int(pts[first - 1]), stream=stream, backward=True, any_frame=False)
                got = 0
                for frame in v.decode(stream):
                    k = position.get(frame.pts)
                    if k is None or k < first:
                        continue
                    if k > last:
                        break
                    frames[:, :, :, 0, breaks[r] + (k - first)] = _frame2array(frame).astype(
                        dt, copy=False
                    )
                    got += 1
                    if k == last:
                        break
                if got != len(run):
                    raise IndexError(f"Frames {first}..{last} could not all be decoded.")

        return ndr_reader_base.selectframeCZ(frames, SelectC, SelectZ)

    # ------------------------------------------------------------------
    # Clock, channels and metadata
    # ------------------------------------------------------------------

    def epochclock(self, epochstreams: list[str] | str, epoch_select: int = 1) -> list[ClockType]:
        """Return ``[ClockType('dev_local_time')]``: a video always has real frame times."""
        return [ClockType("dev_local_time")]

    def t0_t1(self, epochstreams: list[str] | str, epoch_select: int = 1) -> list[list[float]]:
        """Return ``[[firsttime, lasttime]]``, the times of the first and last frames.

        The same convention as ``ndr.reader.tiffstack``. Measured without
        decoding the whole file (see ``endtimes``). ``epoch_select`` is
        accepted for the reader interface and not used: an epoch is one file.
        """
        p = Path(self.videofile(epochstreams))
        return [ndr_reader_video.endtimes(str(p.parent), p.name)]

    def getchannelsepoch(
        self, epochstreams: list[str] | str, epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List the channels of a video epoch: a single ``'image'`` channel ``'image1'``.

        The colour channels of the video are returned together as the C axis
        of ``readframes`` rather than as separate NDR channels.
        """
        return [{"name": "image1", "type": "image", "time_channel": None}]

    def metadata(self, epochstreams: list[str] | str, epoch_select: int = 1) -> dict[str, Any]:
        """Return standardized image-acquisition metadata for a video epoch.

        A video is not a raster scan, so this is the default dict from
        ``emptyimagemetadata`` (``israster=False``) with only ``frame_period``
        filled in, as ``1/FrameRate`` (seconds) from the container header.
        """
        m = ndr_reader_base.emptyimagemetadata()
        with self.videoobject(epochstreams) as v:
            fr = _framerate(_videostream(v))
        if fr > 0:
            m["frame_period"] = 1.0 / fr
        return m

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
            "ndr_reader_video is an image reader; use readframes() instead of "
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
        """Video files carry no native event channels."""
        return np.array([]), np.array([])

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def videoextensions() -> list[str]:
        """Return the file extensions treated as video files."""
        return [
            ".avi",
            ".mp4",
            ".m4v",
            ".mov",
            ".mj2",
            ".mpg",
            ".mpeg",
            ".wmv",
            ".asf",
            ".ogg",
            ".ogv",
            ".mkv",
            ".webm",
        ]

    @staticmethod
    def videonumframes(v) -> int | float:
        """Return the frame count the container reports, or NaN if unavailable."""
        try:
            nf = _videostream(v).frames
        except Exception:
            return float("nan")
        if nf is not None and nf > 0:
            return int(nf)
        return float("nan")

    @staticmethod
    def channelsanddatatype(v) -> tuple[int, str]:
        """Return the colour channel count and class name of the first frame.

        Decodes frame 1 of the open container ``v`` and returns its number of
        colour channels (3 for RGB, 1 for greyscale) and its numeric class
        (e.g. ``'uint8'``). Decoding advances ``v``; callers that read on
        afterwards seek first.
        """
        stream = _videostream(v)
        frame = next(v.decode(stream), None)
        if frame is None:
            raise ValueError(f"{v.name} contains no decodable video frames.")
        a = _frame2array(frame)
        return int(a.shape[2]), a.dtype.name

    @staticmethod
    def checkframeind(frameind: list[int] | np.ndarray, n: int) -> None:
        """Raise ``IndexError`` unless ``frameind`` are integer indices in ``1..n``."""
        f = np.atleast_1d(np.asarray(frameind, dtype=float))
        if np.any(f < 1) or np.any(f > n) or np.any(f != np.round(f)):
            raise IndexError(f"Frame indices must be integers between 1 and {n}.")

    @staticmethod
    def endtimes(folder: str, name: str) -> list[float]:
        """Return ``[first, last]`` frame times of a video file, without decoding it all.

        The presentation times in seconds of the first and last frames of
        ``folder/name``, measured as ``scanframetimes`` measures them, so they
        agree with ``frametimes``. The first is read by decoding frame 1. For
        the last, the reader seeks to a few frame periods before the duration
        and decodes to the end, widening the margin if the seek lands past
        the last frame; only if that fails (no usable frame rate or duration)
        does it fall back to the full scan. ``[NaN, NaN]`` for a file with no
        frames. The result is cached for the session, keyed as
        ``scanframetimes`` keys it.
        """
        filename = str(Path(folder) / name)
        key = _cachekey(filename)
        if key in _ENDTIMES_CACHE:
            return list(_ENDTIMES_CACHE[key])

        with _av().open(filename) as v:
            stream = _videostream(v)
            frame = next(v.decode(stream), None)
            if frame is None:
                t0t1 = [float("nan"), float("nan")]
                _ENDTIMES_CACHE[key] = t0t1
                return list(t0t1)
            first = _frametime(frame, stream)
            last = float("nan")
            fr = _framerate(stream)
            if stream.duration is not None:
                dur = float(stream.duration * stream.time_base)
            elif v.duration is not None:
                dur = v.duration / 1e6  # container duration is in AV_TIME_BASE units
            else:
                dur = float("nan")
            if math.isfinite(fr) and fr > 0 and math.isfinite(dur) and dur > 0:
                margin = 5
                while math.isnan(last):
                    start = max(0.0, dur - margin / fr)
                    target = _origin(stream) + int(start / stream.time_base)
                    v.seek(target, stream=stream, backward=True, any_frame=False)
                    for frame in v.decode(stream):
                        last = _frametime(frame, stream)
                    if start == 0:
                        break  # read from the start: nothing further back to try
                    margin *= 4

        if math.isnan(last):
            last = float(ndr_reader_video.scanframetimes(folder, name)[-1])
        t0t1 = [first, last]
        _ENDTIMES_CACHE[key] = t0t1
        return list(t0t1)

    @staticmethod
    def scanframetimes(folder: str, name: str) -> np.ndarray:
        """Return the per-frame times of a video file, measured by decoding it once.

        Decodes ``folder/name`` frame by frame and records each frame's
        presentation time (seconds from the start of the file). Returns a 1-D
        array with one entry per frame. The result is cached for the session,
        keyed by the full path, byte size and modification time, so a file
        changed on disk is re-scanned.
        """
        return ndr_reader_video._scan(str(Path(folder) / name))[1].copy()

    @staticmethod
    def _scan(filename: str) -> tuple[np.ndarray, np.ndarray]:
        """Decode a video once; return (pts, times) of every frame. Cached."""
        key = _cachekey(filename)
        if key not in _SCAN_CACHE:
            with _av().open(filename) as v:
                stream = _videostream(v)
                fr = _framerate(stream)
                pts = []
                for k, frame in enumerate(v.decode(stream)):
                    if frame.pts is not None:
                        pts.append(frame.pts)
                    elif math.isfinite(fr) and fr > 0:
                        # a frame without a timestamp is placed at its nominal time
                        pts.append(_origin(stream) + round(k / fr / stream.time_base))
                    else:
                        raise ValueError(f"{filename}: frame {k + 1} has no timestamp.")
                pts = np.asarray(pts, dtype=np.int64)
                t = ((pts - _origin(stream)) * float(stream.time_base)).astype(float)
            _SCAN_CACHE[key] = (pts, t)
        return _SCAN_CACHE[key]

    @staticmethod
    def _framepts(filename: str) -> np.ndarray:
        """Presentation-ordered pts of the video packets, read without decoding. Cached.

        Python-only: positions ``readframes``' seeks, the job VideoReader's
        random-access ``read`` does internally in MATLAB.
        """
        key = _cachekey(filename)
        if key not in _PTS_CACHE:
            with _av().open(filename) as v:
                stream = _videostream(v)
                pts = [p.pts for p in v.demux(stream) if p.size > 0]
            if any(p is None for p in pts):
                pts = []  # unusable; readframes falls back to the decoded scan
            _PTS_CACHE[key] = np.sort(np.asarray(pts, dtype=np.int64))
        return _PTS_CACHE[key]
