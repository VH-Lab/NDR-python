"""Read data from a VHLAB LabView (VHLV) ``.vld`` binary file.

Port of +ndr/+format/+vld/readvhlvdatafile.m

The VHLV binary is stored **big-endian** (MATLAB ``'ieee-be'``). Two layouts
are supported, selected by the header ``Multiplexed`` field:

* ``Multiplexed == 1`` (perfectly interleaved): frame 0 holds one sample of
  every channel (ch0, ch1, ..., chN-1), then frame 1, and so on.
* ``Multiplexed == 0`` (chunked): ``SamplesPerChunk`` samples of channel 1 are
  stored, followed by ``SamplesPerChunk`` samples of channel 2, etc., repeating
  per chunk.

When a ``Scale`` field is present, stored integers are converted to physical
units by multiplying by ``Scale/maxint`` (``maxint`` depends on precision).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# precision -> (numpy big-endian dtype, unit size in bytes, maxint for scaling)
# Mirrors the switch in readvhlvdatafile.m / precision2datatype.
_PRECISION: dict[str, tuple[str, int, float]] = {
    "double": (">f8", 8, 1.0),
    "single": (">f4", 4, 1.0),
    "int32": (">i4", 4, float(2**31 - 1)),
    "int16": (">i2", 2, float(2**15 - 1)),
}


def _precision_info(header: dict[str, Any]) -> tuple[str, int, float]:
    """Return (numpy_dtype, unit_size_bytes, maxint) for the header precision."""
    precision = header.get("precision", "double")
    if precision not in _PRECISION:
        raise ValueError(f"Unknown precision {precision} in VHLV header.")
    return _PRECISION[precision]


def total_samples(header: dict[str, Any], myfilename: str) -> int:
    """Estimate the total number of samples per channel in a ``.vld`` file.

    Parameters
    ----------
    header : dict
        Parsed VHLV header (see :func:`readvhlvheaderfile`).
    myfilename : str
        Path to the ``.vld`` data file.

    Returns
    -------
    int
        Number of samples per channel (``bytes / (NumChans * unit_size)``).
    """
    _, unit_size, _ = _precision_info(header)
    if not os.path.exists(myfilename):
        raise ValueError(f"Could not find file {myfilename} to determine its size.")
    nbytes = os.path.getsize(myfilename)
    return nbytes // (int(header["NumChans"]) * unit_size)


def readvhlvdatafile(
    myfilename: str,
    headerstruct: dict[str, Any] | None,
    channelnums: int | list[int] | np.ndarray,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Read data from a VHLV ``.vld`` file.

    Parameters
    ----------
    myfilename : str
        Path to the ``.vld`` file.
    headerstruct : dict or None
        Parsed header. If ``None``, the sibling ``.vlh`` file is read.
    channelnums : int or array-like of int
        Channel numbers to read, 1-based (1 = first acquired channel).
    t0 : float
        Start time (seconds relative to the start of the recording).
    t1 : float
        End time (seconds relative to the start of the recording).

    Returns
    -------
    T : numpy.ndarray
        Sample times, shape ``(n_samples,)``.
    D : numpy.ndarray
        Data, shape ``(n_samples, n_channels)``.
    tot_sam : int
        Estimated total samples per channel in the file.
    tot_time : float
        Estimated total duration of the file, in seconds.
    """
    from ndr.format.vld.readvhlvheaderfile import readvhlvheaderfile

    if headerstruct is None:
        base, _ = os.path.splitext(myfilename)
        headerstruct = readvhlvheaderfile(base + ".vlh")

    channelnums = np.atleast_1d(np.asarray(channelnums, dtype=int))
    num_chans = int(headerstruct["NumChans"])
    if np.any(channelnums < 1) or np.any(channelnums > num_chans):
        raise ValueError(
            "Requested channel numbers must be between 1 and NumChans, which "
            f"for this header file is {num_chans}."
        )
    if t0 < 0:
        raise ValueError("t0 cannot be negative.")
    if t1 < 0:
        raise ValueError("t1 cannot be negative.")

    dtype, unit_size, maxint = _precision_info(headerstruct)
    sr = float(headerstruct["SamplingRate"])

    tot_sam = total_samples(headerstruct, myfilename)
    tot_time = tot_sam / sr

    has_scale = "Scale" in headerstruct
    scale = float(headerstruct["Scale"]) / maxint if has_scale else 1.0

    multiplexed = int(headerstruct.get("Multiplexed", 0))

    if multiplexed:
        T, D = _read_multiplexed(
            myfilename, headerstruct, channelnums, t0, t1, dtype, unit_size, tot_sam
        )
    else:
        T, D = _read_chunked(
            myfilename, headerstruct, channelnums, t0, t1, dtype, unit_size, tot_sam
        )

    if has_scale:
        D = D.astype(np.float64) * scale

    return T, D, tot_sam, tot_time


def _read_multiplexed(
    myfilename: str,
    header: dict[str, Any],
    channelnums: np.ndarray,
    t0: float,
    t1: float,
    dtype: str,
    unit_size: int,
    tot_sam: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read perfectly-interleaved (Multiplexed==1) data. Port of
    readvhlvdatafile_multiplexed."""
    num_chans = int(header["NumChans"])
    sr = float(header["SamplingRate"])

    # samples run 1..N; sample 1 occurs at t==0
    s0 = int(round(1 + t0 * sr))
    s1 = int(min(round(1 + t1 * sr), tot_sam))

    T = np.arange(s0 - 1, s1) / sr
    n = s1 - s0 + 1
    if n <= 0:
        return T, np.zeros((0, len(channelnums)), dtype=np.float64)

    data = np.zeros((n, len(channelnums)), dtype=np.float64)
    itemsize = unit_size
    frame_bytes = num_chans * itemsize
    with open(myfilename, "rb") as fid:
        for c, ch in enumerate(channelnums):
            # start byte of the first requested sample for this channel
            start = ((s0 - 1) * num_chans + (int(ch) - 1)) * itemsize
            fid.seek(start)
            raw = fid.read(frame_bytes * n)
            col = np.frombuffer(raw, dtype=np.dtype(dtype))[::num_chans]
            # guard against a short final read
            data[: len(col), c] = col.astype(np.float64)

    return T, data


def _read_chunked(
    myfilename: str,
    header: dict[str, Any],
    channelnums: np.ndarray,
    t0: float,
    t1: float,
    dtype: str,
    unit_size: int,
    tot_sam: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read chunked (Multiplexed==0) data. Port of the chunk loop in
    readvhlvdatafile.m."""
    num_chans = int(header["NumChans"])
    sr = float(header["SamplingRate"])
    spc = int(header["SamplesPerChunk"])

    s0 = int(round(1 + t0 * sr))
    s1 = int(round(1 + t1 * sr))

    chunkstart = 1 + s0 // spc
    samplesinstartingchunk = s0 % spc
    if samplesinstartingchunk == 0:
        chunkstart -= 1
        samplesinstartingchunk = spc

    chunkstop = 1 + s1 // spc
    samplesinstoppingchunk = s1 % spc
    if samplesinstoppingchunk == 0:
        chunkstop -= 1
        samplesinstoppingchunk = spc

    np_dtype = np.dtype(dtype)
    binary_samples_per_chunk = spc * num_chans * unit_size

    T_parts: list[np.ndarray] = []
    D_parts: list[np.ndarray] = []

    with open(myfilename, "rb") as fid:
        i = chunkstart
        while i <= chunkstop:
            myT = np.arange((i - 1) * spc, i * spc) / sr

            sample_start = samplesinstartingchunk if i == chunkstart else 1
            sample_stop = samplesinstoppingchunk if i == chunkstop else spc

            cols: list[np.ndarray] = []
            hit_eof = False
            for ch in channelnums:
                # seek to the start of channel `ch` within chunk `i`
                offset = (i - 1) * binary_samples_per_chunk + (
                    int(ch) - 1
                ) * spc * unit_size
                fid.seek(offset)
                raw = fid.read(spc * unit_size)
                vals = np.frombuffer(raw, dtype=np_dtype)
                if vals.size == spc:
                    cols.append(vals.astype(np.float64))
                else:
                    hit_eof = True
                    break

            if hit_eof:
                break

            myData = np.column_stack(cols)  # (spc, n_channels)
            # trim to keep only the requested samples (1-based inclusive)
            myData = myData[sample_start - 1 : sample_stop, :]
            myT = myT[sample_start - 1 : sample_stop]

            T_parts.append(myT)
            D_parts.append(myData)
            i += 1

    if T_parts:
        T = np.concatenate(T_parts)
        D = np.concatenate(D_parts, axis=0)
    else:
        T = np.zeros((0,), dtype=np.float64)
        D = np.zeros((0, len(channelnums)), dtype=np.float64)

    return T, D
