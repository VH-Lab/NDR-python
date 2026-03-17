"""Read data from a TDT SEV channel.

Port of +ndr/+format/+tdt/read_SEV_channel.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.tdt.read_SEV_header import _FORMAT_TO_DTYPE, read_SEV_header


def read_SEV_channel(
    dirname: str | Path,
    header: list[dict[str, Any]] | None = None,
    channeltype: str = "ai",
    channel: int = 1,
    s0: int = 1,
    s1: int | float = float("inf"),
) -> np.ndarray:
    """Read data from a single SEV channel.

    Parameters
    ----------
    dirname : str or Path
        Path to the SEV directory.
    header : list of dict or None
        Header information. If None, it will be read.
    channeltype : str
        ``'time'`` or ``'t'`` for timestamps, ``'analog_in'`` or ``'ai'``
        for analog data.
    channel : int
        Channel number to read.
    s0 : int
        First sample to read (1-based).
    s1 : int or float
        Last sample to read (1-based, ``inf`` for end).

    Returns
    -------
    numpy.ndarray
        Data array (column vector).
    """
    dirname = Path(dirname)

    if header is None:
        header = read_SEV_header(dirname)

    # Filter to the requested channel
    chan_entries = [h for h in header if h["chan"] == channel]
    if not chan_entries:
        raise ValueError(f"Channel {channel} not found in {dirname}.")

    # Sort by hour
    chan_entries.sort(key=lambda h: h.get("hour", 0))

    # Compute cumulative sample boundaries
    sample_boundaries = [1]
    for entry in chan_entries:
        sample_boundaries.append(entry["npts"])
    cumulative = np.cumsum(sample_boundaries)

    # Clamp s0, s1
    total = int(cumulative[-1]) - 1
    s0_ = max(1, min(s0, total))
    s1_ = max(1, min(int(s1) if not np.isinf(s1) else total, total))

    if channeltype in ("time", "t"):
        fs = chan_entries[0]["fs"]
        data = np.arange(s0_ - 1, s1_) / fs
        return data.reshape(-1, 1)

    if channeltype not in ("analog_in", "ai"):
        raise ValueError(f"Unknown channeltype '{channeltype}'.")

    # Determine which file blocks we need
    block_0 = int(np.searchsorted(cumulative[1:], s0_, side="left"))
    block_1 = int(np.searchsorted(cumulative[1:], s1_, side="left"))

    data = np.empty(s1_ - s0_ + 1, dtype=np.float64)
    counter = 0

    for block in range(block_0, block_1 + 1):
        entry = chan_entries[block]
        dt = np.dtype(_FORMAT_TO_DTYPE.get(entry["dForm"], np.float32)).newbyteorder("<")

        if block == block_0:
            s0_local = s0_ - int(cumulative[block]) + 1
        else:
            s0_local = 1

        if block == block_1:
            s1_local = s1_ - int(cumulative[block]) + 1
        else:
            s1_local = entry["npts"]

        fpath = dirname / entry["name"]
        with open(fpath, "rb") as fid:
            fid.seek(40 + (s0_local - 1) * entry["itemSize"])
            n = s1_local - s0_local + 1
            d_here = np.fromfile(fid, dtype=dt, count=n)

        data[counter : counter + n] = d_here.astype(np.float64)
        counter += n

    return data.reshape(-1, 1)
