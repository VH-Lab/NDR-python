"""Parse a SpikeGLX .meta file into a standardized header structure.

Port of +ndr/+format/+neuropixelsGLX/header.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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
        - n_neural_chans : int — for NIDQ this is MN + MA + XA
        - n_sync_chans : int — number of digital word int16 columns
        - n_digital_word_cols : int — same as n_sync_chans
        - n_digital_lines : int — number of individual bit lines exposed
        - digital_line_col : ndarray — 0-based DW column offset per line
        - digital_line_bit : ndarray — 0-based bit position per line
        - digital_line_label : list of str — label per line
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

    # Parse snsApLfSy or snsMnMaXaDw to determine neural vs sync channels.
    # Also compute digital line mapping:
    #   n_digital_word_cols : number of int16 columns holding digital data
    #   n_digital_lines     : number of single-bit digital lines exposed
    #   digital_line_col    : (n_digital_lines,) 0-based DW column offset
    #   digital_line_bit    : (n_digital_lines,) 0-based bit position
    #   digital_line_label  : list of str labels per line
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
        # IMEC sync: each sync column provides 16 bits
        info["n_digital_word_cols"] = info["n_sync_chans"]
        n_lines = 16 * info["n_sync_chans"]
        info["n_digital_lines"] = n_lines
        cols = np.zeros(n_lines, dtype=int)
        bits = np.zeros(n_lines, dtype=int)
        labels: list[str] = []
        idx = 0
        for c in range(info["n_sync_chans"]):
            for b in range(16):
                cols[idx] = c
                bits[idx] = b
                labels.append(f"SY{c}.{b}")
                idx += 1
        info["digital_line_col"] = cols
        info["digital_line_bit"] = bits
        info["digital_line_label"] = labels
    elif "snsMnMaXaDw" in meta:
        info["stream_type"] = "nidq"
        counts = [int(x) for x in meta["snsMnMaXaDw"].split(",")]
        info["n_mn_chans"] = counts[0]  # multiplexed neural
        info["n_ma_chans"] = counts[1]  # multiplexed analog
        info["n_xa_chans"] = counts[2]  # non-multiplexed analog
        info["n_dw_chans"] = counts[3]  # digital word int16 columns
        info["n_neural_chans"] = counts[0] + counts[1] + counts[2]
        info["n_sync_chans"] = counts[3]
        info["n_digital_word_cols"] = counts[3]

        # Bytes saved per port — each byte = 8 active digital lines
        n_bytes_p0 = int(meta.get("niXDBytes1", "0"))
        n_bytes_p1 = int(meta.get("niXDBytes2", "0"))

        if n_bytes_p0 == 0 and n_bytes_p1 == 0:
            # Fall back: assume all 16 bits of every DW column are active
            n_lines_p0 = 16 * info["n_dw_chans"]
            n_lines_p1 = 0
        else:
            n_lines_p0 = 8 * n_bytes_p0
            n_lines_p1 = 8 * n_bytes_p1

        n_lines = n_lines_p0 + n_lines_p1
        info["n_digital_lines"] = n_lines
        cols = np.zeros(n_lines, dtype=int)
        bits = np.zeros(n_lines, dtype=int)
        labels = []
        idx = 0
        for k in range(n_lines_p0):
            abs_bit = k
            cols[idx] = abs_bit // 16
            bits[idx] = abs_bit % 16
            labels.append(f"XD{k}")
            idx += 1
        for k in range(n_lines_p1):
            abs_bit = n_bytes_p0 * 8 + k
            cols[idx] = abs_bit // 16
            bits[idx] = abs_bit % 16
            labels.append(f"XD1.{k}")
            idx += 1
        info["digital_line_col"] = cols
        info["digital_line_bit"] = bits
        info["digital_line_label"] = labels
    else:
        info["stream_type"] = "unknown"
        info["n_neural_chans"] = info["n_saved_chans"] - 1
        info["n_sync_chans"] = 1
        info["n_digital_word_cols"] = 1
        info["n_digital_lines"] = 16
        info["digital_line_col"] = np.zeros(16, dtype=int)
        info["digital_line_bit"] = np.arange(16, dtype=int)
        info["digital_line_label"] = [f"bit{b}" for b in range(16)]

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
    elif "niAiRangeMax" in meta:
        vmax = float(meta["niAiRangeMax"])
        vmin = float(meta["niAiRangeMin"])
        info["voltage_range"] = (vmin, vmax)
    else:
        info["voltage_range"] = (-0.6, 0.6)  # Neuropixels 1.0 default

    # Max integer value
    if "imMaxInt" in meta:
        info["max_int"] = int(meta["imMaxInt"])
    elif "niMaxInt" in meta:
        info["max_int"] = int(meta["niMaxInt"])
    else:
        info["max_int"] = 512  # Neuropixels 1.0 default

    # NI-DAQ gains
    if "niMNGain" in meta:
        info["ni_mn_gain"] = float(meta["niMNGain"])
    if "niMAGain" in meta:
        info["ni_ma_gain"] = float(meta["niMAGain"])

    # Bits per sample
    info["bits_per_sample"] = 16

    # File size and duration
    info["file_size_bytes"] = int(meta.get("fileSizeBytes", "0"))
    info["file_time_secs"] = float(meta.get("fileTimeSecs", "0"))

    # Probe information
    info["probe_type"] = meta.get("imDatPrb_type", "")
    info["probe_sn"] = meta.get("imDatPrb_sn", "")

    return info
