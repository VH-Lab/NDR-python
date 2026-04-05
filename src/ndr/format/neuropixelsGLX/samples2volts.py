"""Convert raw int16 samples to voltage in volts.

Port of +ndr/+format/+neuropixelsGLX/samples2volts.m
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def _parse_imro_gains(imroTbl_str: str, stream_type: str) -> list[float]:
    """Extract per-channel gains from imroTbl string.

    The imroTbl format varies by probe type but generally:
        (probeType,nChan)(chanIdx bank refIdx apGain lfGain apHiPass)
    AP gain is at position 4 and LF gain at position 5 (1-indexed
    within each channel entry).
    """
    gains: list[float] = []

    tokens = re.findall(r"\(([^)]+)\)", imroTbl_str)
    if len(tokens) < 2:
        return gains

    # First token is the header, skip it
    for token in tokens[1:]:
        vals = token.replace(",", " ").split()
        int_vals = [int(v) for v in vals if v.lstrip("-").isdigit()]
        if len(int_vals) >= 5:
            if stream_type.lower() == "ap":
                gains.append(float(int_vals[3]))
            else:
                gains.append(float(int_vals[4]))
        elif len(int_vals) >= 4:
            gains.append(float(int_vals[3]))

    return gains


def samples2volts(data: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    """Convert raw int16 samples to voltage in volts.

    The conversion formula for Neuropixels imec channels is::

        volts = double(data) * imAiRangeMax / imMaxInt / gain

    Per-channel gains are parsed from the imroTbl field. If imroTbl is
    absent, default gains are used (500 for AP, 250 for LF).

    Parameters
    ----------
    data : ndarray
        N x C matrix of raw int16 samples (neural channels only).
    info : dict
        Header dictionary from :func:`ndr.format.neuropixelsGLX.header`.

    Returns
    -------
    ndarray
        N x C matrix of voltages in volts (float64).
    """
    vmax = info["voltage_range"][1]  # imAiRangeMax
    meta = info["meta"]

    # Parse per-channel gains from imroTbl if available
    if "imroTbl" in meta:
        gains = _parse_imro_gains(meta["imroTbl"], info["stream_type"])
        n_chans = data.shape[1] if data.ndim > 1 else 1
        if len(gains) >= n_chans:
            gains = gains[:n_chans]
        else:
            gains = gains + [gains[-1]] * (n_chans - len(gains))

        gains_arr = np.array(gains, dtype=np.float64)
        # Official formula: V = int16 * imAiRangeMax / imMaxInt / gain
        volts = data.astype(np.float64) * (vmax / (info["max_int"] * gains_arr))
    else:
        # Default gain for Neuropixels 1.0
        if info["stream_type"].lower() == "ap":
            default_gain = 500.0
        else:
            default_gain = 250.0
        volts = data.astype(np.float64) * vmax / (info["max_int"] * default_gain)

    return volts
