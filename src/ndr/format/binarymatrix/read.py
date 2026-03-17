"""Read data from a binary matrix file.

Port of +ndr/+format/+binarymatrix/read.m
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_DTYPE_MAP = {
    "double": np.float64,
    "single": np.float32,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _bytes_per_value(data_type: str) -> int:
    """Return the number of bytes per value for the given data type."""
    if data_type in _DTYPE_MAP:
        return np.dtype(_DTYPE_MAP[data_type]).itemsize
    m = re.search(r"\d+", data_type)
    if m:
        return int(m.group()) // 8
    raise ValueError(f"Cannot determine bytes per value for dataType '{data_type}'.")


def read(
    filename_or_fileobj: str | Path,
    num_channels: int,
    channel_indexes: int | list[int] | np.ndarray,
    s0: float,
    s1: float,
    *,
    dataType: str = "double",
    byteOrder: str = "ieee-le",
    force_single_channel_read: bool = False,
    headerSkip: int = 0,
) -> tuple[np.ndarray, int, int, int]:
    """Read data from a binary matrix file.

    A binary matrix file is comprised of a vector of channels.  Each sample
    consists of one value for each channel, followed by the next sample, etc.

    Parameters
    ----------
    filename_or_fileobj : str or Path
        Path to the binary file.
    num_channels : int
        Number of channels that comprise the sample vector.
    channel_indexes : int, list of int, or ndarray
        1-based channel indexes to return.
    s0 : float
        Starting sample number (1-based). ``-inf`` means the first sample.
    s1 : float
        Ending sample number (1-based). ``inf`` means the last sample.
    dataType : str
        Data type of each value (e.g. ``'double'``, ``'single'``, ``'int16'``).
    byteOrder : str
        ``'ieee-le'`` for little-endian or ``'ieee-be'`` for big-endian.
    force_single_channel_read : bool
        Force reading channels one at a time.
    headerSkip : int
        Number of header bytes to skip.

    Returns
    -------
    data : ndarray
        SxC matrix with samples in rows and channels in columns.
    total_samples : int
        Total number of vector samples in the file.
    s0 : int
        The actual starting sample number.
    s1 : int
        The actual ending sample number.
    """
    filename = Path(filename_or_fileobj)
    filesize = filename.stat().st_size

    if isinstance(channel_indexes, (int, np.integer)):
        channel_indexes = [int(channel_indexes)]
    channel_indexes = np.asarray(channel_indexes, dtype=np.int64).ravel()

    if np.any(channel_indexes > num_channels) or np.any(channel_indexes < 1):
        raise IndexError(f"channel_indexes out of range; must be 1..{num_channels}")

    if byteOrder not in ("ieee-le", "ieee-be"):
        raise ValueError(f"byteOrder must be 'ieee-le' or 'ieee-be', got '{byteOrder}'")

    bpv = _bytes_per_value(dataType)
    total_samples = (filesize - headerSkip) // (num_channels * bpv)

    s0 = int(max(1, s0)) if not (math.isinf(s0) and s0 < 0) else 1
    s1 = int(min(total_samples, s1)) if not (math.isinf(s1) and s1 > 0) else total_samples

    bytes_per_sample = bpv * num_channels

    dt = _DTYPE_MAP.get(dataType)
    if dt is None:
        raise ValueError(f"Unsupported dataType '{dataType}'")
    if byteOrder == "ieee-be":
        dt = np.dtype(dt).newbyteorder(">")
    else:
        dt = np.dtype(dt).newbyteorder("<")

    # Sort channels for potential consecutive optimization
    chan_sort_idx = np.argsort(channel_indexes)
    chan_sorted = channel_indexes[chan_sort_idx]

    consecutive = len(channel_indexes) == 1 or np.all(np.diff(chan_sorted) == 1)

    num_samples = s1 - s0 + 1

    with open(filename, "rb") as f:
        if not force_single_channel_read and consecutive:
            chs_before = int(chan_sorted[0]) - 1
            chs_after = num_channels - int(chan_sorted[-1])
            n_ch = len(channel_indexes)

            offset = headerSkip + (s0 - 1) * bytes_per_sample + chs_before * bpv
            skip_after = (chs_after + chs_before) * bpv

            f.seek(offset)
            if skip_after == 0:
                raw = np.fromfile(f, dtype=dt, count=n_ch * num_samples)
            else:
                # Read with skipping: read n_ch values, skip skip_after bytes, repeat
                raw = np.empty(n_ch * num_samples, dtype=dt)
                for i in range(num_samples):
                    chunk = np.frombuffer(f.read(n_ch * bpv), dtype=dt)
                    raw[i * n_ch : (i + 1) * n_ch] = chunk
                    if i < num_samples - 1:
                        f.seek(skip_after, 1)

            data = raw.reshape(num_samples, n_ch)
            # Un-sort if needed
            if not np.array_equal(chan_sorted, channel_indexes):
                unsort = np.empty_like(chan_sort_idx)
                unsort[chan_sort_idx] = np.arange(len(chan_sort_idx))
                data = data[:, unsort]
        else:
            data = np.empty((num_samples, len(channel_indexes)), dtype=dt)
            for c_idx, ch in enumerate(channel_indexes):
                chs_before = int(ch) - 1
                chs_after = num_channels - int(ch)
                offset = headerSkip + (s0 - 1) * bytes_per_sample + chs_before * bpv
                skip_after = (chs_after + chs_before) * bpv

                f.seek(offset)
                col = np.empty(num_samples, dtype=dt)
                for i in range(num_samples):
                    col[i] = np.frombuffer(f.read(bpv), dtype=dt)[0]
                    if i < num_samples - 1:
                        f.seek(skip_after, 1)
                data[:, c_idx] = col

    # Cast to requested precision
    data = data.astype(dt)

    return data, total_samples, s0, s1
