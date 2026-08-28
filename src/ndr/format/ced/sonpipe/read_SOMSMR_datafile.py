"""Read samples, event times, or markers through the sonpipe CLI.

Port of +ndr/+format/+ced/+sonpipe/read_SOMSMR_datafile.m
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.ced.sonpipe._invoke import invoke_binary, invoke_json
from ndr.format.ced.sonpipe.channelinfo import channelinfo
from ndr.format.ced.sonpipe.read_SOMSMR_header import read_SOMSMR_header

# CED channel kinds, as in sonpipe.channels.
WAVEFORM_KINDS = (1, 9)  # Adc, RealWave
EVENT_KINDS = (2, 3, 4)  # EventFall, EventRise, EventBoth
MARKER_KINDS = (5, 6, 7, 8)  # Marker, AdcMark, RealMark, TextMark


def _timewindow(t0: float, t1: float) -> list[str]:
    """Mirror timewindow() in the MATLAB port: only pass finite, useful bounds."""
    args: list[str] = []
    if math.isfinite(t0) and t0 > 0:
        args += ["--t0", f"{t0:.12g}"]
    if math.isfinite(t1):
        args += ["--t1", f"{t1:.12g}"]
    return args


def _markers_to_output(r: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    markers = r.get("markers") or []
    if not markers:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    time = np.array([float(m["time"]) for m in markers], dtype=np.float64)
    codes = np.array([m.get("code", 0) for m in markers], dtype=np.float64)
    return time, codes


def read_SOMSMR_datafile(
    filename: str | Path,
    header: dict[str, Any] | None = None,
    channel_number: int = 1,
    t0: float = 0.0,
    t1: float = float("inf"),
) -> tuple[np.ndarray, float | None, float | None, Any, np.ndarray | None]:
    """Read one channel.

    Returns ``(data, total_samples, total_time, blockinfo, time)``, mirroring the
    MATLAB signature. ``blockinfo`` is always None here: it is a sigTOOL concept
    with no sonpipe equivalent, and the MATLAB sonpipe port leaves it empty too.

    Only one channel may be read per call, as in the MATLAB version.
    """
    if header is None:
        header = read_SOMSMR_header(filename)

    if not np.isscalar(channel_number) and len(np.atleast_1d(channel_number)) > 1:
        raise ValueError("Only one channel may be read per call; channel_number must be scalar.")

    info = channelinfo(header, int(channel_number))
    kind = int(info["kind"])
    total_time = info.get("max_time")
    total_samples: float | None = None
    blockinfo = None

    base = ["read", str(filename), "-c", str(int(channel_number))]

    if kind in WAVEFORM_KINDS:
        sr = info.get("samplerate")
        total_samples = info.get("num_samples")
        if sr is None or not total_samples:
            return np.array([]), total_samples, total_time, blockinfo, np.array([])

        if t0 < 0 or math.isinf(t0):
            t0 = 0.0
        s0 = max(0, int(math.floor(t0 * sr)))
        s1 = int(total_samples) - 1 if math.isinf(t1) else int(math.floor(t1 * sr))
        count = s1 - s0 + 1
        if count <= 0:
            return np.array([]), total_samples, total_time, blockinfo, np.array([])

        data = invoke_binary([*base, "--start", str(s0), "--count", str(count)], "double")
        time = (s0 + np.arange(data.size, dtype=np.float64)) / sr
        return data, total_samples, total_time, blockinfo, time

    if kind in EVENT_KINDS:
        data = invoke_binary([*base, *_timewindow(t0, t1)], "double")
        return data, total_samples, total_time, blockinfo, data

    if kind in MARKER_KINDS:
        r = invoke_json([*base, *_timewindow(t0, t1), "--json"])
        time, data = _markers_to_output(r)
        return data, total_samples, total_time, blockinfo, time

    raise ValueError(f"Unsupported CED channel kind {kind} for channel {channel_number}.")
