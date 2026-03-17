"""Read data from an NDR text signal file.

Port of +ndr/+format/+textSignal/readData.m
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.textSignal.readHeader import readHeader


def _parse_time(s: str, time_units: str) -> float:
    """Parse a time string to a float (POSIX time for datestamp, or numeric)."""
    if time_units == "datestamp":
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return float(s)


def _read_and_sort_events(filename: Path, time_units: str):
    """Read all events from a text signal file and sort by time."""
    raw_events: dict[str, list] = {}
    file_times: list[float] = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                channel = int(parts[0])
            except ValueError:
                continue

            event: dict[str, Any] = {
                "time": _parse_time(parts[1], time_units),
                "command": parts[2].upper(),
                "value1": float("nan"),
                "value2": float("nan"),
                "endtime": float("nan"),
            }
            file_times.append(event["time"])

            if event["command"] == "SET" and len(parts) >= 4:
                event["value1"] = float(parts[3])
            elif event["command"] == "RAMP" and len(parts) >= 6:
                event["value1"] = float(parts[3])
                event["value2"] = float(parts[4])
                event["endtime"] = _parse_time(parts[5], time_units)
                file_times.append(event["endtime"])

            ch_key = f"ch{channel}"
            if ch_key not in raw_events:
                raw_events[ch_key] = []
            raw_events[ch_key].append(event)

    # Sort events by time
    events = {}
    for ch_key, ev_list in raw_events.items():
        events[ch_key] = sorted(ev_list, key=lambda e: e["time"])

    if file_times:
        return events, min(file_times), max(file_times)
    return events, float("-inf"), float("inf")


def readData(
    filename: str | Path,
    channels: list[int],
    t0: float = float("-inf"),
    t1: float = float("inf"),
    *,
    dT: float = float("nan"),
    timestamps: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Read data from an NDR text signal file.

    Parameters
    ----------
    filename : str or Path
        Path to the text signal file.
    channels : list of int
        Channel numbers to read.
    t0 : float
        Start time. ``-inf`` starts from the beginning.
    t1 : float
        End time. ``inf`` reads to the end.
    dT : float
        If not NaN, resample with this time step.
    timestamps : ndarray or None
        If provided, evaluate at these specific timestamps.

    Returns
    -------
    D : list of ndarray
        Data for each channel.
    T : ndarray
        Time vector.
    """
    filename = Path(filename)
    header = readHeader(filename)
    events, file_t_start, file_t_end = _read_and_sort_events(filename, header["time_units"])

    if math.isinf(t0) and t0 < 0:
        t0 = file_t_start
    if math.isinf(t1) and t1 > 0:
        t1 = file_t_end

    # Generate time vector
    if not math.isnan(dT):
        T = np.arange(t0, t1 + dT * 0.5, dT)
    elif timestamps is not None:
        T = np.asarray(timestamps, dtype=np.float64)
    else:
        all_times = {t0, t1}
        for ch_num in channels:
            ch_key = f"ch{ch_num}"
            if ch_key in events:
                for ev in events[ch_key]:
                    if t0 <= ev["time"] <= t1:
                        all_times.add(ev["time"])
                    if ev["command"] == "RAMP" and t0 <= ev["endtime"] <= t1:
                        all_times.add(ev["endtime"])
        T = np.array(sorted(all_times))
        T = T[(T >= t0) & (T <= t1)]

    # Evaluate signal for each channel
    D = []
    for ch_num in channels:
        ch_key = f"ch{ch_num}"
        values = np.zeros(len(T), dtype=np.float64)
        if ch_key in events:
            ch_events = events[ch_key]
            for j, t in enumerate(T):
                # Find last event at or before time t
                last_idx = -1
                for k, ev in enumerate(ch_events):
                    if ev["time"] <= t:
                        last_idx = k
                    else:
                        break

                if last_idx >= 0:
                    ev = ch_events[last_idx]
                    if ev["command"] == "SET":
                        values[j] = ev["value1"]
                    elif ev["command"] == "RAMP":
                        if ev["time"] <= t <= ev["endtime"]:
                            duration = ev["endtime"] - ev["time"]
                            if duration > 0:
                                frac = (t - ev["time"]) / duration
                                values[j] = ev["value1"] + frac * (ev["value2"] - ev["value1"])
                            else:
                                values[j] = ev["value2"]
                        else:
                            values[j] = ev["value2"]
                    elif ev["command"] == "NONE":
                        # Look back for the last non-NONE event
                        prev_val = 0.0
                        for k in range(last_idx - 1, -1, -1):
                            prev_ev = ch_events[k]
                            if prev_ev["command"] == "SET":
                                prev_val = prev_ev["value1"]
                                break
                            elif prev_ev["command"] == "RAMP":
                                prev_val = prev_ev["value2"]
                                break
                        values[j] = prev_val
        D.append(values)

    return D, T
