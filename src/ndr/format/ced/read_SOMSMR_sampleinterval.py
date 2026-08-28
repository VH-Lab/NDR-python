"""Read the sample interval of one CED Spike2 channel.

Port of +ndr/+format/+ced/read_SOMSMR_sampleinterval.m

See read_SOMSMR_header for why every CED read here goes through sonpipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ndr.format.ced.sonpipe.read_SOMSMR_sampleinterval import (
    read_SOMSMR_sampleinterval as _sonpipe_read_SOMSMR_sampleinterval,
)


def read_SOMSMR_sampleinterval(
    filename: str | Path,
    header: dict[str, Any] | None = None,
    channel_number: int = 1,
) -> tuple[float, float, float]:
    """Return ``(sampleinterval, total_samples, total_time)`` in seconds.

    ``header`` is accepted and ignored, mirroring the MATLAB signature.

    This returns three values, matching MATLAB. The previous neo-backed version
    returned a fourth, the neo reader object; nothing consumed it and there is
    no sonpipe equivalent, so it is gone.
    """
    return _sonpipe_read_SOMSMR_sampleinterval(filename, header, channel_number)
