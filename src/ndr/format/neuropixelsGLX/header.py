"""Parse a SpikeGLX .meta file into a standardized header structure.

Port of +ndr/+format/+neuropixelsGLX/header.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ndr.format.neuropixelsGLX.readmeta import readmeta


def _parse_channel_subset(subset_str: str, n_saved_chans: int) -> list[int]:
    """Parse the snsSaveChanSubset field.

    The field can be 'all' or a comma-separated list of 0-based indices
    and ranges (e.g. '0:383,768'). Returns a 1-based channel index list.
    """
    if subset_str.strip().lower() == "all":
        return list(range(1, n_saved_chans + 1))

    chan_list: list[int] = []
    parts = subset_str.strip().split(",")
    for part in parts:
        part = part.strip()
        if ":" in part:
            range_vals = part.split(":")
            start, end = int(range_vals[0]), int(range_vals[1])
            chan_list.extend(range(start, end + 1))
        else:
            chan_list.append(int(part))

    # Convert from 0-based to 1-based
    return [c + 1 for c in chan_list]


def header(metafilename: str | Path) -> dict[str, Any]:
    """Parse a SpikeGLX .meta file into a standardized header dictionary.

    Parameters
    ----------
    metafilename : str or Path
        Full path to the .meta file.

    Returns
    -------
    dict
        Dictionary with fields:
        - sample_rate : float
        - n_saved_chans : int
        - n_neural_chans : int
        - n_sync_chans : int
        - saved_chan_list : list of int (1-based)
        - voltage_range : tuple of (float, float)
        - max_int : int
        - bits_per_sample : int
        - file_size_bytes : int
        - file_time_secs : float
        - probe_type : str
        - probe_sn : str
        - stream_type : str ('ap', 'lf', 'nidq', or 'unknown')
        - meta : dict (raw metadata)
    """
    metafilename = Path(metafilename)
    meta = readmeta(metafilename)

    info: dict[str, Any] = {}
    info["meta"] = meta

    # Sample rate
    if "imSampRate" in meta:
        info["sample_rate"] = float(meta["imSampRate"])
    elif "niSampRate" in meta:
        info["sample_rate"] = float(meta["niSampRate"])
    else:
        raise ValueError("Could not find sample rate in meta file.")

    # Number of saved channels
    info["n_saved_chans"] = int(meta["nSavedChans"])

    # Parse snsApLfSy or snsMnMaXaDw to determine neural vs sync channels
    if "snsApLfSy" in meta:
        counts = [int(x) for x in meta["snsApLfSy"].split(",")]
        # counts[0] = AP chans, counts[1] = LF chans, counts[2] = SY chans
        info["n_sync_chans"] = counts[2]
        # Determine if this is AP or LF from filename
        name = metafilename.stem
        if ".lf" in name:
            info["stream_type"] = "lf"
            info["n_neural_chans"] = counts[1]
        else:
            info["stream_type"] = "ap"
            info["n_neural_chans"] = counts[0]
    elif "snsMnMaXaDw" in meta:
        info["stream_type"] = "nidq"
        counts = [int(x) for x in meta["snsMnMaXaDw"].split(",")]
        info["n_neural_chans"] = counts[0]
        info["n_sync_chans"] = 0
    else:
        info["stream_type"] = "unknown"
        info["n_neural_chans"] = info["n_saved_chans"] - 1
        info["n_sync_chans"] = 1

    # Parse saved channel subset
    if "snsSaveChanSubset" in meta:
        info["saved_chan_list"] = _parse_channel_subset(
            meta["snsSaveChanSubset"], info["n_saved_chans"]
        )
    else:
        info["saved_chan_list"] = list(range(1, info["n_saved_chans"] + 1))

    # Voltage range
    if "imAiRangeMax" in meta:
        vmax = float(meta["imAiRangeMax"])
        vmin = float(meta["imAiRangeMin"])
        info["voltage_range"] = (vmin, vmax)
    else:
        info["voltage_range"] = (-0.6, 0.6)  # Neuropixels 1.0 default

    # Max integer value
    if "imMaxInt" in meta:
        info["max_int"] = int(meta["imMaxInt"])
    else:
        info["max_int"] = 512  # Neuropixels 1.0 default

    # Bits per sample
    info["bits_per_sample"] = 16

    # File size and duration
    info["file_size_bytes"] = int(meta.get("fileSizeBytes", "0"))
    info["file_time_secs"] = float(meta.get("fileTimeSecs", "0"))

    # Probe information
    info["probe_type"] = meta.get("imDatPrb_type", "")
    info["probe_sn"] = meta.get("imDatPrb_sn", "")

    return info
