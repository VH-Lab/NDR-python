"""Read samples, event times, or markers from a CED Spike2 file.

Port of +ndr/+format/+ced/read_SOMSMR_datafile.m

See read_SOMSMR_header for why every CED read here goes through sonpipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.ced.sonpipe.read_SOMSMR_datafile import (
    read_SOMSMR_datafile as _sonpipe_read_SOMSMR_datafile,
)


def read_SOMSMR_datafile(
    filename: str | Path,
    header: dict[str, Any] | None = None,
    channel_number: int = 1,
    t0: float = 0.0,
    t1: float = float("inf"),
) -> tuple[np.ndarray, float | None, float | None, Any, np.ndarray | None]:
    """Read one channel, returning ``(data, total_samples, total_time, blockinfo, time)``.

    ``data`` and ``time`` are column vectors, as elsewhere in NDR-python.
    ``blockinfo`` is always None: it is a sigTOOL concept with no sonpipe
    equivalent, and the MATLAB sonpipe port leaves it empty too.
    """
    data, total_samples, total_time, blockinfo, time = _sonpipe_read_SOMSMR_datafile(
        filename, header, channel_number, t0, t1
    )
    data = np.asarray(data).reshape(-1, 1)
    time = None if time is None else np.asarray(time).reshape(-1, 1)
    return data, total_samples, total_time, blockinfo, time
