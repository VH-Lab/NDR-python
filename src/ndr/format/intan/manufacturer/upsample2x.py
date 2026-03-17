"""Upsample waveforms by a factor of 2 using cubic spline interpolation.

Port of +ndr/+format/+intan/+manufacturer/upsample2x.m
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def upsample2x(
    t: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample sampled voltages by a factor of 2 using cubic spline interpolation.

    Parameters
    ----------
    t : numpy.ndarray
        1-D time vector of shape ``(num_samples,)``.
    v : numpy.ndarray
        Voltage array of shape ``(num_channels, num_samples)``.

    Returns
    -------
    t2 : numpy.ndarray
        New time vector with 2x the number of samples.
    v2 : numpy.ndarray
        Interpolated voltage array of shape ``(num_channels, 2*num_samples)``.
    """
    num_channels, num_samples = v.shape
    sample_rate = 1.0 / (t[1] - t[0])

    # Build the 2x time vector by interleaving original and midpoints
    t_mid = t + 1.0 / (2.0 * sample_rate)
    t2 = np.empty(2 * num_samples, dtype=t.dtype)
    t2[0::2] = t
    t2[1::2] = t_mid

    v2 = np.zeros((num_channels, 2 * num_samples), dtype=v.dtype)
    for i in range(num_channels):
        cs = CubicSpline(t, v[i, :])
        v2[i, :] = cs(t2)

    return t2, v2
