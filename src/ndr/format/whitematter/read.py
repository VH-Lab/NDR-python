"""Read data from a WhiteMatter LLC binary data file.

Port of +ndr/+format/+whitematter/read.m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ndr.format.binarymatrix.read import read as binarymatrix_read
from ndr.time.fun.samples2times import samples2times
from ndr.time.fun.times2samples import times2samples


def read(
    fname: str | Path,
    t0: float = 0.0,
    t1: float = float("inf"),
    *,
    numChans: int = 64,
    SR: float = 20000,
    byteOrder: str = "ieee-le",
    channels: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Read data from a WhiteMatter LLC binary data file.

    Parameters
    ----------
    fname : str or Path
        Path to the ``.bin`` file.
    t0 : float
        Start time in seconds (``-inf`` for beginning).
    t1 : float
        End time in seconds (``inf`` for end).
    numChans : int
        Total number of interleaved channels.
    SR : float
        Sampling rate in Hz.
    byteOrder : str
        ``'ieee-le'`` or ``'ieee-be'``.
    channels : list of int or None
        1-based channel indices to read. None reads all channels.

    Returns
    -------
    D : ndarray
        N x C data matrix (int16).
    t : ndarray
        Time vector in seconds.
    t0_t1 : list of float
        ``[start_time, end_time]`` of the full file.
    """
    fname = Path(fname)
    data_type = "int16"
    bytes_per_value = 2
    header_skip = 8

    filesize = fname.stat().st_size
    total_data_bytes = filesize - header_skip
    total_samples = total_data_bytes // (bytes_per_value * numChans)

    file_t_start = 0.0
    file_t_end = float(samples2times(np.array([total_samples]), [0.0, 0.0], SR)[0])
    t0_t1 = [file_t_start, file_t_end]

    if channels is None:
        channels_to_read = list(range(1, numChans + 1))
    else:
        channels_to_read = list(channels)

    t0_req = max(t0, t0_t1[0])
    t1_req = min(t1, t0_t1[1])

    if t1_req < t0_req:
        return np.array([], dtype=np.int16), np.array([]), t0_t1

    s0_req = int(times2samples(np.array([t0_req]), t0_t1, SR)[0])
    s1_req = int(times2samples(np.array([t1_req]), t0_t1, SR)[0])
    s0_req = max(1, s0_req)
    s1_req = min(total_samples, s1_req)

    if s1_req < s0_req:
        return np.array([], dtype=np.int16), np.array([]), t0_t1

    D, _, s0_actual, s1_actual = binarymatrix_read(
        fname,
        numChans,
        channels_to_read,
        float(s0_req),
        float(s1_req),
        dataType=data_type,
        byteOrder=byteOrder,
        headerSkip=header_skip,
    )

    t = samples2times(np.arange(s0_actual, s1_actual + 1), t0_t1, SR)

    return D, t, t0_t1
