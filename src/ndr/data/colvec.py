"""Column vector utility.

Port of +ndr/+data/colvec.m
"""

from __future__ import annotations

import numpy as np


def colvec(x: np.ndarray | list) -> np.ndarray:
    """Reshape an array into a column vector.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    numpy.ndarray
        Column vector (N, 1) shape.
    """
    x = np.asarray(x)
    return x.reshape(-1, 1)
