"""Enumerate SmartSPIM channels under an acquisition root.

Port of ``+ndr/+format/+smartspim/listChannels.m``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from pydantic import validate_call

from ndr.format.smartspim.readAcquisitionMetadata import readAcquisitionMetadata

__all__ = ["listChannels"]

_CHANNEL_RE = re.compile(r"^Ex_(\d+)_Em_(.+)_Ch(\d+)$")


def _laser_key(wavelength: float) -> str:
    return f"w{round(float(wavelength))}"


def _parse_channel_name(name: str) -> tuple[float, str, float]:
    m = _CHANNEL_RE.match(name)
    if not m:
        raise ValueError(f'Cannot parse channel name "{name}"; ' 'expected "Ex_<n>_Em_<f>_Ch<n>".')
    return float(m.group(1)), m.group(2), float(m.group(3))


@validate_call
def listChannels(rootDir: str) -> list[dict[str, Any]]:
    """Enumerate the channel directories under ``rootDir``.

    Returns one dict per channel with fields:

    - ``name``: directory name (e.g. ``"Ex_561_Em_561F_Ch2"``)
    - ``wavelength``: excitation wavelength (nm)
    - ``filterName``: emission filter identifier (e.g. ``"561F"``)
    - ``filterChannel``: hardware filter channel number
    - ``laserPowerLeft``: left-side laser power percent, NaN if unknown
    - ``laserPowerRight``: right-side laser power percent, NaN if unknown
    - ``numTiles``: tile count from ``metadata.json`` (NaN if metadata.json
      does not enumerate tiles)

    Discovery walks the filesystem for directories matching the naming
    convention. metadata.json is consulted for laser power and per-channel
    tile counts; missing metadata is tolerated. Sorted alphabetically by
    name for a stable listing.
    """
    if not rootDir:
        raise ValueError("rootDir must be a non-empty string.")
    root = Path(rootDir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {rootDir}")

    names = sorted(e.name for e in root.iterdir() if e.is_dir() and _CHANNEL_RE.match(e.name))
    if not names:
        raise ValueError(f"No channel directories (Ex_*_Em_*_Ch*) found in {rootDir}")

    laser_by_wavelength: dict[str, dict[str, float]] = {}
    num_tiles_by_name: dict[str, int] = {}
    meta_path = root / "metadata.json"
    if meta_path.is_file():
        try:
            meta = readAcquisitionMetadata(rootDir)
            for lp in meta["laserPower"]:
                if math.isnan(lp["wavelength"]):
                    continue
                laser_by_wavelength[_laser_key(lp["wavelength"])] = lp
            for t in meta["tiles"]:
                cn = t["channelName"]
                if not cn:
                    continue
                num_tiles_by_name[cn] = num_tiles_by_name.get(cn, 0) + 1
        except (OSError, ValueError):
            laser_by_wavelength = {}
            num_tiles_by_name = {}

    channels: list[dict[str, Any]] = []
    for name in names:
        wavelength, filter_name, filter_channel = _parse_channel_name(name)
        key = _laser_key(wavelength)
        lp = laser_by_wavelength.get(key)
        if lp is not None:
            left_pct = lp["leftPct"]
            right_pct = lp["rightPct"]
        else:
            left_pct = float("nan")
            right_pct = float("nan")
        n = float(num_tiles_by_name.get(name, float("nan")))
        channels.append(
            {
                "name": name,
                "wavelength": wavelength,
                "filterName": filter_name,
                "filterChannel": filter_channel,
                "laserPowerLeft": left_pct,
                "laserPowerRight": right_pct,
                "numTiles": n,
            }
        )
    return channels
