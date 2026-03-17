"""Read header information from an Axon Instruments ABF file.

Port of +ndr/+format/+axon/read_abf_header.m

This module requires the ``pyabf`` package for reading ABF files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pyabf
except ImportError:  # pragma: no cover
    pyabf = None  # type: ignore[assignment]


def read_abf_header(filename: str | Path) -> dict[str, Any]:
    """Read header information from an ABF file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.abf`` file.

    Returns
    -------
    dict
        Header information including sample interval, channel names,
        recording times, etc.
    """
    if pyabf is None:
        raise ImportError(
            "pyabf is required for reading ABF files. Install with: pip install pyabf"
        )

    abf = pyabf.ABF(str(filename), loadData=False)

    header: dict[str, Any] = {}
    header["si"] = 1e6 / abf.dataRate  # sample interval in microseconds
    header["recChNames"] = abf.adcNames
    header["recChUnits"] = abf.adcUnits
    header["nADCNumChannels"] = abf.channelCount
    header["lActualAcqLength"] = abf.dataPointCount
    header["recTime"] = [0.0, abf.dataLengthSec]
    header["lActualEpisodes"] = abf.sweepCount
    header["nOperationMode"] = abf.abfFileHeader.nOperationMode if hasattr(abf, "abfFileHeader") else 0

    if abf.sweepCount > 1:
        header["sweepLengthInPts"] = abf.sweepPointCount
        sweep_starts = []
        for i in range(abf.sweepCount):
            sweep_starts.append(i * abf.sweepPointCount)
        header["sweepStartInPts"] = sweep_starts

    return header
