"""Read one channel's sample interval through the sonpipe CLI.

Port of +ndr/+format/+ced/+sonpipe/read_SOMSMR_sampleinterval.m
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _nan_if_none(value: Any) -> float:
    """MATLAB's emptytonan: a null from the CLI means "not applicable"."""
    return math.nan if value is None else float(value)


def read_SOMSMR_sampleinterval(
    filename: str | Path,
    header: dict[str, Any] | None = None,  # noqa: ARG001 - mirrors the MATLAB signature
    channel_number: int = 1,
) -> tuple[float, float, float]:
    """Return ``(sampleinterval, total_samples, total_time)`` in seconds.

    ``header`` is accepted and ignored, as in the MATLAB version: the CLI reads
    the file itself.
    """
    from ndr.format.ced.sonpipe._invoke import invoke_json

    r = invoke_json(["sampleinterval", str(filename), "-c", str(int(channel_number))])
    return (
        _nan_if_none(r.get("sampleinterval")),
        _nan_if_none(r.get("total_samples")),
        _nan_if_none(r.get("total_time")),
    )
