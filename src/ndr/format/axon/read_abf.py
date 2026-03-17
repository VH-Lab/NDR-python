"""Read data from an Axon Instruments ABF file.

Port of +ndr/+format/+axon/read_abf.m

This module requires the ``pyabf`` package for reading ABF files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyabf
except ImportError:  # pragma: no cover
    pyabf = None  # type: ignore[assignment]

from ndr.format.axon.read_abf_header import read_abf_header


def read_abf(
    filename: str | Path,
    header: dict[str, Any] | None = None,
    channel_type: str = "ai",
    channel_numbers: int | list[int] = 1,
    t0: float = 0.0,
    t1: float = float("inf"),
) -> np.ndarray:
    """Read data from an Axon Instruments ABF file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.abf`` file.
    header : dict or None
        Header information. If None, it will be read from the file.
    channel_type : str
        ``'ai'`` for analog input or ``'time'`` for timestamps.
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
    if pyabf is None:
        raise ImportError(
            "pyabf is required for reading ABF files. Install with: pip install pyabf"
        )

    filename = str(filename)

    if header is None:
        header = read_abf_header(filename)

    if isinstance(channel_numbers, int):
        channel_numbers = [channel_numbers]

    si_sec = header["si"] * 1e-6  # sample interval in seconds

    if t0 < 0:
        t0 = 0.0
    max_time = header["recTime"][1] - header["recTime"][0] - si_sec
    if t1 > max_time:
        t1 = max_time

    channel_type_lower = channel_type.lower()

    if channel_type_lower == "time":
        if "sweepLengthInPts" not in header:
            data = np.arange(t0, t1 + si_sec * 0.5, si_sec)
        else:
            all_times = np.array([], dtype=np.float64)
            for start_pts in header["sweepStartInPts"]:
                start_time = start_pts * si_sec
                end_time = start_time + (header["sweepLengthInPts"] - 1) * si_sec
                sweep_times = np.arange(start_time, end_time + si_sec * 0.5, si_sec)
                all_times = np.concatenate([all_times, sweep_times])
            mask = (all_times >= t0) & (all_times <= t1 + 0.5 * si_sec)
            data = all_times[mask]
        return data.reshape(-1, 1)

    elif channel_type_lower in ("ai", "analog_in"):
        abf = pyabf.ABF(filename)

        s0 = int(round(t0 / si_sec))
        s1_idx = int(round(t1 / si_sec)) + 1

        columns = []
        for ch_num in sorted(channel_numbers):
            ch_idx = ch_num - 1  # 0-based
            if abf.sweepCount == 1:
                abf.setSweep(0, channel=ch_idx)
                col = abf.sweepY[s0:s1_idx].copy()
            else:
                # Multi-sweep: concatenate all sweeps then slice
                all_data = np.array([], dtype=np.float32)
                for sweep in range(abf.sweepCount):
                    abf.setSweep(sweep, channel=ch_idx)
                    all_data = np.concatenate([all_data, abf.sweepY])
                # Apply time filter
                times = np.arange(len(all_data)) * si_sec
                mask = (times >= t0) & (times <= t1 + 0.5 * si_sec)
                col = all_data[mask]
            columns.append(col)

        data = np.column_stack(columns) if columns else np.array([])
        return data

    else:
        raise ValueError(f"Unknown channel type '{channel_type}'")
