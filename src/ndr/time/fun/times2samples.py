"""Convert sample times to sample index numbers.

Port of +ndr/+time/+fun/times2samples.m
"""

from __future__ import annotations

import numpy as np


def matlab_round(x: np.ndarray | float) -> np.ndarray:
    """Round half away from zero, matching MATLAB's ``round``.

    numpy's ``round`` and Python's builtin ``round`` round halves to even
    (banker's rounding); MATLAB rounds halves away from zero. On exact
    half-sample boundaries the two disagree, so the sample index returned by a
    read differs between the languages. Use this at every sample-boundary
    computation to keep Python and MATLAB in lockstep.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


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
    s = np.asarray(1 + matlab_round((t - t0_t1[0]) * sr), dtype=float)

    # Handle -inf times -> sample 1
    neg_inf = np.isinf(t) & (t < 0)
    s[neg_inf] = 1

    # Handle +inf times -> last sample
    pos_inf = np.isinf(t) & (t > 0)
    s[pos_inf] = 1 + sr * (t0_t1[1] - t0_t1[0])

    return s
