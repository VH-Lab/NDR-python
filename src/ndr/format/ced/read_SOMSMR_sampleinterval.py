"""Read sample interval from a CED SOM/SMR file.

Port of +ndr/+format/+ced/read_SOMSMR_sampleinterval.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from neo.rawio.cedrawio import CedRawIO
except ImportError:  # pragma: no cover
    CedRawIO = None  # type: ignore[assignment]

from ndr.format.ced.read_SOMSMR_header import read_SOMSMR_header


def read_SOMSMR_sampleinterval(
    filename: str | Path,
    header: dict[str, Any] | None = None,
    channel_number: int = 1,
) -> tuple[float, int | None, float | None, Any]:
    """Read the sample interval for a channel from a CED SOM/SMR file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.smr`` or ``.son`` file.
    header : dict or None
        Header information. If None, it will be read from the file.
    channel_number : int
        Channel number for which to return the sample interval.

    Returns
    -------
    sampleinterval : float
        Sample interval in seconds (``NaN`` for event channels).
    total_samples : int or None
        Total sample count.
    total_time : float or None
        Total duration in seconds.
    blockinfo : object
        Block information.
    """
    if CedRawIO is None:
        raise ImportError("neo is required for reading CED files. Install with: pip install neo")

    filename = Path(filename)

    if header is None:
        header = read_SOMSMR_header(filename)

    raw_reader = CedRawIO(filename=str(filename))
    raw_reader.parse_header()

    sig_channels = raw_reader.header["signal_channels"]
    ch_ids = sig_channels["id"]

    ch_idx = None
    for idx, cid in enumerate(ch_ids):
        if int(cid) == channel_number:
            ch_idx = idx
            break

    if ch_idx is not None:
        sr = float(sig_channels[ch_idx]["sampling_rate"])
        sampleinterval = 1.0 / sr
        n_samples = raw_reader.get_signal_size(block_index=0, seg_index=0)
        total_samples = n_samples
        total_time = n_samples / sr
    else:
        # Event or marker channel
        sampleinterval = float("nan")
        total_samples = None
        total_time = None

    return sampleinterval, total_samples, total_time, raw_reader
