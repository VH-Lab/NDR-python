"""Convert sample times to sample index numbers.

Port of +ndr/+time/+fun/times2samples.m
"""

from __future__ import annotations

import numpy as np


def matlab_round(x: np.ndarray | float) -> np.ndarray | float:
    """Round half away from zero, the way MATLAB's ``round`` does.

    Python's builtin ``round`` and ``numpy.round`` both round halves to even
    (banker's rounding), so round(0.5) is 0 and round(2.5) is 2. MATLAB rounds
    halves away from zero: 1 and 3. Every time-to-sample conversion in NDR-matlab
    goes through MATLAB ``round``, so on an exact half-sample boundary the two
    ports otherwise disagree by one sample -- reading a different span of the
    file for the same requested interval.

    Returns a float for scalar input and an ndarray for array input, so callers
    can keep wrapping the result in ``int()`` as they did with the builtin.
    """
    a = np.asarray(x, dtype=float)
    r = np.where(a >= 0, np.floor(a + 0.5), np.ceil(a - 0.5))
    return float(r) if np.isscalar(x) or a.ndim == 0 else r


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
