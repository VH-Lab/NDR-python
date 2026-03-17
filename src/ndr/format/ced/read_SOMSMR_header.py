"""Read header information from a CED SOM/SMR file.

Port of +ndr/+format/+ced/read_SOMSMR_header.m

This module requires the ``sonpy`` or ``neo`` package for reading CED files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import neo
    from neo.rawio.cedrawio import CedRawIO
except ImportError:  # pragma: no cover
    neo = None  # type: ignore[assignment]
    CedRawIO = None  # type: ignore[assignment]


def read_SOMSMR_header(filename: str | Path) -> dict[str, Any]:
    """Read header information from a CED SOM/SMR file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.smr`` or ``.son`` file.

    Returns
    -------
    dict
        Header with ``fileinfo`` and ``channelinfo`` sub-dicts.
    """
    if neo is None:
        raise ImportError("neo is required for reading CED files. Install with: pip install neo")

    filename = Path(filename)
    ext = filename.suffix.lower()

    if ext not in (".smr", ".son"):
        raise ValueError(f"Unknown extension for SOM/SMR file: {ext}")

    raw_reader = CedRawIO(filename=str(filename))
    raw_reader.parse_header()

    header: dict[str, Any] = {}
    header["fileinfo"] = {
        "filename": filename.name,
    }

    channelinfo = []
    sig_channels = raw_reader.header.get("signal_channels", [])
    for ch in sig_channels:
        info = dict(zip(sig_channels.dtype.names, ch))
        info["kind"] = 1  # ADC
        channelinfo.append(info)

    event_channels = raw_reader.header.get("event_channels", [])
    for ch in event_channels:
        info = dict(zip(event_channels.dtype.names, ch))
        info["kind"] = 2  # event
        channelinfo.append(info)

    header["channelinfo"] = channelinfo
    return header
