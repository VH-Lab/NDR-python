"""Read pixels from an OME-Zarr pyramid.

Port of ``+ndr/+format/+omezarr/readArray.m``.

MATLAB has no zarr library, so the MATLAB side reads chunks and decodes
Blosc/Zstd itself via ``private/*.m``. Python has the ``zarr`` library, so
those private helpers collapse into a single ``zarr.open()`` call. See
this package's ``ndr_matlab_python_bridge.yaml`` for the ``not_applicable``
mapping of every ``private/*.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import validate_call

from ndr.format.omezarr.resolveArrayPath import resolveArrayPath

__all__ = ["readArray"]


def _zarr():
    """Import zarr lazily so it is only required when a store is read."""
    try:
        import zarr
    except ImportError as err:  # pragma: no cover - exercised without zarr
        raise ImportError(
            "zarr is required for reading OME-Zarr stores. "
            'Install with: pip install "ndr[formats]"'
        ) from err
    return zarr


def _validate_region(region: Any, shape: tuple[int, ...]) -> tuple[list[int], list[int]]:
    nd = len(shape)
    region = np.asarray(region)
    if region.shape != (2, nd):
        raise ValueError(
            f"Region must be a 2-by-{nd} numeric matrix [start; stop] "
            "(one column per axis in shape order)."
        )
    region_start = [int(v) for v in region[0, :]]
    region_stop = [int(v) for v in region[1, :]]
    if any(v < 1 for v in region_start) or any(v != float(int(v)) for v in region[0, :]):
        raise ValueError("Region start values must be positive integers.")
    if any(a < b for a, b in zip(region_stop, region_start)):
        raise ValueError("Region stop must be >= start on every axis.")
    if any(a > b for a, b in zip(region_stop, list(shape))):
        raise ValueError(f"Region stop {region_stop} exceeds array shape {list(shape)}.")
    return region_start, region_stop


@validate_call(config={"arbitrary_types_allowed": True})
def readArray(
    zarrPath: str,
    pyramidName: str,
    level: int,
    *,
    Region: Any = None,
    OutputType: str = "",
) -> np.ndarray:
    """Return one region of one pyramid level as an n-D numpy array.

    Positional inputs:

    - ``zarrPath``: directory containing ``.zattrs`` (the OME-Zarr store root)
    - ``pyramidName``: one of the names returned by
      :func:`ndr.format.omezarr.listPyramids`. There is no default: pyramid
      selection is always explicit, because that is the silent-failure guard
      issue #127 named.
    - ``level``: 1-based level number. Level 1 is the highest resolution
      (the first entry in the pyramid's ``datasets`` array).

    Options:

    - ``Region``: 2-by-N numeric matrix ``[start; stop]``, each column one
      axis, in the array's stored axis order (typically c, z, y, x). All
      values are 1-based and INCLUSIVE, so ``[1; shape]`` reads the whole
      array along that axis. Default: the whole array.
    - ``OutputType``: cast the returned array to this MATLAB class name
      (e.g. ``'uint16'``, ``'single'``). Default: '' (return the array's
      native type).

    The MATLAB reader is limited to a hand-rolled Blosc/Zstd path because
    MATLAB has no zarr library; the Python reader delegates to the
    ``zarr`` library, which handles v2, v3, all supported codecs and
    byte orders.
    """
    array_dir = resolveArrayPath(zarrPath, pyramidName, level)
    zarr_mod = _zarr()

    z = zarr_mod.open(array_dir, mode="r")
    shape = tuple(int(x) for x in z.shape)
    nd = len(shape)

    if (
        Region is None
        or (hasattr(Region, "size") and Region.size == 0)
        or (isinstance(Region, (list, tuple)) and len(Region) == 0)
    ):
        region_start = [1] * nd
        region_stop = list(shape)
    else:
        region_start, region_stop = _validate_region(Region, shape)

    slicer = tuple(slice(s - 1, e) for s, e in zip(region_start, region_stop))
    data = np.asarray(z[slicer])

    # zarr already handles the fill value + shape correctly. If the region
    # was degenerate (stop < start on any axis), give back an empty array
    # with the underlying dtype so callers can still reason about shape.
    out_shape = tuple(e - s + 1 for s, e in zip(region_start, region_stop))
    if any(v < 1 for v in out_shape):
        data = np.zeros(out_shape, dtype=data.dtype)

    if OutputType:
        # Map MATLAB class names to numpy dtypes (single/double are the
        # only ones that differ).
        matlab_to_numpy = {"single": "float32", "double": "float64"}
        target = matlab_to_numpy.get(OutputType, OutputType)
        data = data.astype(np.dtype(target))

    return data
