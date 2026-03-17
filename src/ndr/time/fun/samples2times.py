"""Convert sample index numbers to sample times.

Port of +ndr/+time/+fun/samples2times.m
"""

from __future__ import annotations

import numpy as np


def samples2times(
    s: np.ndarray | int | float,
    t0_t1: tuple[float, float] | list[float],
    sr: float,
) -> np.ndarray:
    """Convert sample index numbers to sample times.

    Parameters
    ----------
    s : array-like
        Sample index numbers (1-based, matching MATLAB convention).
    t0_t1 : tuple of (t0, t1)
        The beginning and end times of the recording.
    sr : float
        The fixed sample rate in Hz.

    Returns
    -------
    numpy.ndarray
        Times corresponding to each sample.
    """
    s = np.asarray(s, dtype=float)
    t = (s - 1) / sr + t0_t1[0]

    # Handle -inf samples -> t0
    neg_inf = np.isinf(s) & (s < 0)
    t[neg_inf] = t0_t1[0]

    # Handle +inf samples -> t1
    pos_inf = np.isinf(s) & (s > 0)
    t[pos_inf] = t0_t1[1]

    return t
