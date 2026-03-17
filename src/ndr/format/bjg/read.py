"""Read data from a BJG file.

Port of +ndr/+format/+bjg/read.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.binarymatrix.read import read as binarymatrix_read
from ndr.format.bjg.read_bjg_header import read_bjg_header
from ndr.time.fun.times2samples import times2samples


def read(
    filename: str | Path,
    header: dict[str, Any] | None = None,
    channel_type: str = "ai",
    channel_numbers: int | list[int] = 1,
    t0: float = 0.0,
    t1: float = float("inf"),
) -> np.ndarray:
    """Read data from a BJG file.

    BJG files contain float32/single data channels.

    Parameters
    ----------
    filename : str or Path
        Path to the BJG file.
    header : dict or None
        Header data. If None, it will be read from the file.
    channel_type : str
        ``'time'`` or ``'ti'`` for timestamps, ``'ai'`` for analog input.
    channel_numbers : int or list of int
        Channel numbers to read (1-based).
    t0 : float
        Start time in seconds.
    t1 : float
        End time in seconds.

    Returns
    -------
    numpy.ndarray
        Data array.
    """
    filename = Path(filename)

    if header is None:
        header = read_bjg_header(filename)

    if isinstance(channel_numbers, int):
        channel_numbers = [channel_numbers]

    T0_file = header["local_t0"]
    T1_file = header["local_t1"]

    if t0 < 0 or t0 == float("-inf"):
        t0 = T0_file
    if t1 > T1_file:
        t1 = T1_file

    if channel_type in ("time", "ti"):
        data = np.arange(t0, t1 + 0.5 / header["sample_rate"], 1.0 / header["sample_rate"])
        return data.reshape(-1, 1)

    S = times2samples(np.array([t0, t1]), np.array([T0_file, T1_file]), header["sample_rate"])

    data, _, _, _ = binarymatrix_read(
        filename,
        header["num_channels"],
        channel_numbers,
        S[0],
        S[1],
        headerSkip=header["header_size"],
        byteOrder="ieee-le",
        dataType="single",
    )

    return data
