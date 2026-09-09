"""Block-reduce an array for pyramid building.

Python mirror of ``+ndr/+format/+omezarr/reduce.m``. Only supports
'mean' and 'max'; those are the reductions first-class in the
lightsheet NGFF layout the lab writes.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np


def reduce(
    arr: np.ndarray,
    factor: Union[int, Sequence[int]],
    reduction: str,
) -> np.ndarray:
    """Downsample ``arr`` by ``factor`` using ``reduction``.

    Parameters
    ----------
    arr
        The input array. Must be numeric.
    factor
        Scalar downsampling factor applied to every axis, or one
        per-axis factor aligned with ``arr.shape``.
    reduction
        ``"mean"`` or ``"max"``.

    Returns
    -------
    numpy.ndarray
        The reduced array. dtype is preserved for ``max``; ``mean``
        returns ``float64``.

    Notes
    -----
    Trailing partial blocks are kept (an axis of length 5 with factor
    2 produces 3 samples: blocks of 2, 2, 1). This matches how the
    lab's NGFF writers size coarser levels.
    """
    if reduction not in ("mean", "max"):
        raise ValueError(f"reduction must be 'mean' or 'max'; got {reduction!r}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"reduce needs a numeric array; got dtype {arr.dtype}")

    nd = arr.ndim
    if np.isscalar(factor):
        factors = [int(factor)] * nd
    else:
        factors = [int(f) for f in factor]
        if len(factors) != nd:
            raise ValueError(
                f"factor must be scalar or one per axis; got {len(factors)} for a {nd}-D array"
            )
    if any(f < 1 for f in factors):
        raise ValueError("factor entries must be positive integers")

    shape = arr.shape
    out_shape = tuple(-(-shape[d] // factors[d]) for d in range(nd))  # ceil div

    if reduction == "mean":
        out = np.zeros(out_shape, dtype=np.float64)
    else:
        out = np.zeros(out_shape, dtype=arr.dtype)

    # Loop-based; correct and easy to trust for the small ranks used by
    # existing writers (factor 2 across z/y/x). Swap to a strided view
    # for performance when a benchmark says to.
    it = np.ndindex(out_shape)
    for coord in it:
        slices = tuple(
            slice(coord[d] * factors[d], min(shape[d], (coord[d] + 1) * factors[d]))
            for d in range(nd)
        )
        block = arr[slices]
        if reduction == "mean":
            out[coord] = float(np.mean(block))
        else:
            out[coord] = np.max(block)

    return out


__all__ = ["reduce"]
