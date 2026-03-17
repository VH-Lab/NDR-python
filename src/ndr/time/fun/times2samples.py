"""Convert sample times to sample index numbers.

Port of +ndr/+time/+fun/times2samples.m
"""

from __future__ import annotations

import numpy as np


def times2samples(
    t: np.ndarray | float,
    t0_t1: tuple[float, float] | list[float],
    sr: float,
) -> np.ndarray:
    """Convert sample times to sample index numbers.

    Parameters
    ----------
    t : array-like
        Times of samples in seconds.
    t0_t1 : tuple of (t0, t1)
        The beginning and end times of the recording.
    sr : float
        The fixed sample rate in Hz.

    Returns
    -------
    numpy.ndarray
        Sample index numbers (1-based, matching MATLAB convention).
    """
    t = np.asarray(t, dtype=float)
    s = 1 + np.round((t - t0_t1[0]) * sr)

    # Handle -inf times -> sample 1
    neg_inf = np.isinf(t) & (t < 0)
    s[neg_inf] = 1

    # Handle +inf times -> last sample
    pos_inf = np.isinf(t) & (t > 0)
    s[pos_inf] = 1 + sr * (t0_t1[1] - t0_t1[0])

    return s
