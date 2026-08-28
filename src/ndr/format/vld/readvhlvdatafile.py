"""Read VHLV (VH Lab LabView) binary data files.

Port of +ndr/+format/+vld/readvhlvdatafile.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.vld.readvhlvheaderfile import readvhlvheaderfile
from ndr.time.fun.times2samples import matlab_round

# MATLAB precision name -> (numpy big-endian dtype, unit size in bytes, maxint)
_PRECISION = {
    "double": (np.dtype(">f8"), 8, 1),
    "single": (np.dtype(">f4"), 4, 1),
    "int32": (np.dtype(">i4"), 4, 2**31 - 1),
    "int16": (np.dtype(">i2"), 2, 2**15 - 1),
}


def _precision_info(headerstruct: dict[str, Any]) -> tuple[np.dtype, int, int, str]:
    """Return (dtype, unit_size, maxint, output_dtype_name) for a header."""
    precision = headerstruct.get("precision", "double")
    if precision not in _PRECISION:
        raise ValueError(f"Unknown precision {precision} in VHLV header.")
    dtype, unit_size, maxint = _PRECISION[precision]

    output_precision = precision
    if "Scale" in headerstruct and precision in ("int16", "int32"):
        # Scaling an integer read produces a real-valued result.
        output_precision = "single"

    return dtype, unit_size, maxint, output_precision


def _output_dtype(name: str) -> np.dtype:
    return {
        "double": np.dtype(np.float64),
        "single": np.dtype(np.float32),
        "int32": np.dtype(np.int32),
        "int16": np.dtype(np.int16),
    }[name]


def _read_strided(fid, offset: int, count: int, dtype: np.dtype, stride_bytes: int) -> np.ndarray:
    """Read ``count`` values of ``dtype`` starting at ``offset``, ``stride_bytes`` apart.

    ``stride_bytes`` is the gap between the start of consecutive values, so a
    stride equal to the item size is a plain contiguous read. Mirrors MATLAB's
    ``fread(fid, count, precision, skip)``. Returns fewer than ``count`` values
    if the file ends first.

    In this format the stride is always a whole number of items (``unit_size``
    for chunked data, ``unit_size * NumChans`` for multiplexed), so the values
    can be read contiguously and sliced.
    """
    itemsize = dtype.itemsize
    if count <= 0:
        return np.array([], dtype=dtype)
    if stride_bytes % itemsize:
        raise ValueError(
            f"stride of {stride_bytes} bytes is not a multiple of the {itemsize}-byte item size."
        )

    step = stride_bytes // itemsize
    span_items = (count - 1) * step + 1

    fid.seek(offset)
    raw = fid.read(span_items * itemsize)
    values = np.frombuffer(raw, dtype=dtype, count=len(raw) // itemsize)
    return values[::step] if step > 1 else values


def readvhlvdatafile(
    myfilename: str | Path,
    headerstruct: dict[str, Any] | None,
    channelnums: int | list[int] | np.ndarray,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Read LabView data from the VH Lab (VHLV) format.

    Reads data from the multichannel VHLab LabView binary data file format.

    Parameters
    ----------
    myfilename : str or Path
        The data file to open (extension ``.vld``).
    headerstruct : dict or None
        The header as returned by
        :func:`ndr.format.vld.readvhlvheaderfile`. If ``None``, a file with
        the same name but extension ``.vlh`` is opened as the header.
    channelnums : int or list of int
        The channel numbers to read, where 1 is the first channel acquired in
        LabView, 2 the second, and so on. See the header to learn the mapping
        between the channel list and the device inputs (``ai0``, ``ai1``, ...).
    t0 : float
        Time relative to the beginning of the recording to start reading from.
    t1 : float
        Time relative to the beginning of the recording to read to.

    Returns
    -------
    T : numpy.ndarray
        The time of each sample.
    D : numpy.ndarray
        The data, one row per sample and one column per channel.
    tot_sam : float
        The estimated total number of samples in the file.
    tot_time : float
        The estimated time length of the file.

    Notes
    -----
    The data can be stored in two binary layouts. If the header's
    ``Multiplexed`` field is absent or 0, the data are stored in chunks with
    ``SamplesPerChunk`` samples of channel 1, then ``SamplesPerChunk`` samples
    of channel 2, and so on. If ``Multiplexed`` is 1, the samples are
    interleaved so that sample 1 is the first sample of channel 1, sample 2
    the first sample of channel 2, and so on.
    """
    myfilename = Path(myfilename)

    if headerstruct is None or not headerstruct:
        headerstruct = readvhlvheaderfile(myfilename.with_suffix(".vlh"))

    if isinstance(channelnums, (int, np.integer)):
        channelnums = [int(channelnums)]
    channelnums = [int(c) for c in np.asarray(channelnums).ravel()]

    num_chans = int(headerstruct["NumChans"])
    if any(c < 1 or c > num_chans for c in channelnums):
        raise ValueError(
            "Requested channel numbers must be between 1 and NumChans, "
            f"which for this header file is {num_chans}."
        )

    if t0 < 0:
        raise ValueError("t0 cannot be negative.")
    if t1 < 0:
        raise ValueError("t1 cannot be negative.")

    sr = float(headerstruct["SamplingRate"])
    samples_per_chunk = int(headerstruct["SamplesPerChunk"])
    dtype, unit_size, maxint, output_precision = _precision_info(headerstruct)
    out_dtype = _output_dtype(output_precision)
    scale = headerstruct.get("Scale", None)
    multiplexed = bool(headerstruct.get("Multiplexed", 0))

    tot_sam = myfilename.stat().st_size / (num_chans * unit_size)
    tot_time = tot_sam / sr

    # Samples run from 1...N; sample 1 occurs at t == 0.
    s0 = int(matlab_round(1 + t0 * sr))
    s1 = int(matlab_round(1 + t1 * sr))

    with open(myfilename, "rb") as fid:
        if multiplexed:
            D, T = _read_multiplexed(
                fid,
                num_chans,
                sr,
                channelnums,
                s0,
                s1,
                dtype,
                unit_size,
                maxint,
                out_dtype,
                scale,
                tot_sam,
            )
            return T, D, tot_sam, tot_time

        return (
            *_read_chunked(
                fid,
                num_chans,
                sr,
                samples_per_chunk,
                channelnums,
                s0,
                s1,
                dtype,
                unit_size,
                maxint,
                out_dtype,
                scale,
            ),
            tot_sam,
            tot_time,
        )


def _apply_scale(
    values: np.ndarray,
    out_dtype: np.dtype,
    scale: float | None,
    maxint: int,
) -> np.ndarray:
    if scale is None:
        return values
    return (values.astype(out_dtype) * (float(scale) / maxint)).astype(out_dtype)


def _read_multiplexed(
    fid,
    num_chans: int,
    sr: float,
    channels: list[int],
    s0: int,
    s1: int,
    dtype: np.dtype,
    unit_size: int,
    maxint: int,
    out_dtype: np.dtype,
    scale: float | None,
    tot_sam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read perfectly multiplexed (sample-interleaved) data."""
    s1 = min(s1, int(tot_sam))
    n = s1 - s0 + 1
    t = np.arange(s0 - 1, s1) / sr

    data = np.zeros((max(n, 0), len(channels)), dtype=out_dtype)
    if n <= 0:
        return data, t

    stride = unit_size * num_chans
    for c, ch in enumerate(channels):
        offset = ((s0 - 1) * num_chans + (ch - 1)) * unit_size
        col = _read_strided(fid, offset, n, dtype, stride)
        col = _apply_scale(col, out_dtype, scale, maxint)
        data[: len(col), c] = col[:n]

    return data, t


def _read_chunked(
    fid,
    num_chans: int,
    sr: float,
    samples_per_chunk: int,
    channels: list[int],
    s0: int,
    s1: int,
    dtype: np.dtype,
    unit_size: int,
    maxint: int,
    out_dtype: np.dtype,
    scale: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read chunked data: SamplesPerChunk of channel 1, then channel 2, ..."""
    chunkstart = 1 + s0 // samples_per_chunk
    samplesinstartingchunk = s0 % samples_per_chunk
    if samplesinstartingchunk == 0:
        # A modulus of exactly 0 means the last sample of the previous chunk.
        chunkstart -= 1
        samplesinstartingchunk = samples_per_chunk

    chunkstop = 1 + s1 // samples_per_chunk
    samplesinstoppingchunk = s1 % samples_per_chunk
    if samplesinstoppingchunk == 0:
        chunkstop -= 1
        samplesinstoppingchunk = samples_per_chunk

    binary_samples_per_chunk = samples_per_chunk * num_chans * unit_size

    T_parts: list[np.ndarray] = []
    D_parts: list[np.ndarray] = []

    i = chunkstart
    while i <= chunkstop:
        my_t = np.arange((i - 1) * samples_per_chunk, i * samples_per_chunk) / sr

        sample_start = samplesinstartingchunk if i == chunkstart else 1
        sample_stop = samplesinstoppingchunk if i == chunkstop else samples_per_chunk

        columns = []
        short_read = False
        for ch in channels:
            offset = (i - 1) * binary_samples_per_chunk + (ch - 1) * samples_per_chunk * unit_size
            col = _read_strided(fid, offset, samples_per_chunk, dtype, unit_size)
            if len(col) != samples_per_chunk:
                # We hit the end of the file; discard this partial chunk.
                short_read = True
                break
            columns.append(_apply_scale(col, out_dtype, scale, maxint))

        if short_read:
            break

        my_data = np.column_stack(columns) if columns else np.empty((samples_per_chunk, 0))
        # Trim to just the samples wanted from this chunk (1-based inclusive).
        my_data = my_data[sample_start - 1 : sample_stop, :]
        my_t = my_t[sample_start - 1 : sample_stop]

        T_parts.append(my_t)
        D_parts.append(my_data)
        i += 1

    if not D_parts:
        return np.array([]), np.zeros((0, len(channels)), dtype=out_dtype)

    return np.concatenate(T_parts), np.concatenate(D_parts, axis=0)
