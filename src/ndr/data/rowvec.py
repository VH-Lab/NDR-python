"""Row vector utility.

Port of +ndr/+data/rowvec.m
"""

from __future__ import annotations

import numpy as np


def rowvec(x: np.ndarray | list) -> np.ndarray:
    """Reshape an array into a row vector.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    numpy.ndarray
        Row vector (1, N) shape.
    """
    x = np.asarray(x)
    return x.reshape(1, -1)
