"""Cheap metadata probe of an OME-Zarr store.

Python mirror of ``+ndr/+format/+omezarr/probe.m``. Use this from an
ingest / GUI confirmation screen that needs "should we accept this?"
info before the user commits. For anything that must distinguish
pyramids or reach deeper levels, use :func:`listPyramids` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ndr.format.omezarr.isOMEZarr import isOMEZarr
from ndr.format.omezarr.listPyramids import listPyramids


@dataclass
class OMEZarrProbe:
    """Result of :func:`probe`.

    ``ok`` is False when the directory does not parse as an OME-Zarr
    store; every other field is then at its default.
    """

    ok: bool = False
    n_pyramids: int = 0
    pyramid_names: list[str] = field(default_factory=list)
    pyramid_types: list[str] = field(default_factory=list)
    axes_order: str = ""
    axes_units: list[str] = field(default_factory=list)
    level0_shape: list[int] = field(default_factory=list)
    level0_chunks: list[int] = field(default_factory=list)
    level0_scale: list[float] = field(default_factory=list)
    dtype: str = ""
    n_levels_per_pyramid: list[int] = field(default_factory=list)


def probe(zarr_path: str) -> OMEZarrProbe:
    """Metadata-only summary of an OME-Zarr store."""
    out = OMEZarrProbe()

    if not os.path.isdir(zarr_path):
        return out
    try:
        if not isOMEZarr(zarr_path):
            return out
    except Exception:
        return out

    pyramids = listPyramids(zarr_path)
    if not pyramids:
        return out

    out.ok = True
    out.n_pyramids = len(pyramids)
    out.pyramid_names = [str(p.get("name", "") or "") for p in pyramids]
    out.pyramid_types = [str(p.get("type", "") or "") for p in pyramids]

    axes = pyramids[0].get("axes") or []
    out.axes_order = "".join(str(a.get("name", "") or "") for a in axes)
    out.axes_units = [str(a.get("unit", "") or "") for a in axes]

    levels = pyramids[0].get("levels") or []
    if levels:
        L0 = levels[0]
        out.level0_shape = list(L0.get("shape") or [])
        out.level0_chunks = list(L0.get("chunks") or [])
        out.level0_scale = list(L0.get("scale") or [])
        out.dtype = str(L0.get("dtype") or "")

    out.n_levels_per_pyramid = [len(p.get("levels") or []) for p in pyramids]
    return out


__all__ = ["OMEZarrProbe", "probe"]
