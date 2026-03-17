"""Read header information from an NDR text signal file.

Port of +ndr/+format/+textSignal/readHeader.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def readHeader(filename: str | Path) -> dict[str, Any]:
    """Read the header of an NDR text signal file.

    Parameters
    ----------
    filename : str or Path
        Path to the text signal file.

    Returns
    -------
    dict
        Header with ``num_channels`` and ``time_units`` fields.
    """
    filename = Path(filename)

    channels = set()
    time_units = ""

    _datestamp_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            try:
                channel = int(parts[0])
                channels.add(channel)
            except ValueError:
                continue

            if not time_units:
                if _datestamp_re.match(parts[1]):
                    time_units = "datestamp"
                else:
                    time_units = "numeric"

    return {
        "num_channels": len(channels),
        "time_units": time_units,
    }
