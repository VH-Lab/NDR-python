"""Read header information from a WhiteMatter LLC binary filename.

Port of +ndr/+format/+whitematter/header.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def header(filename: str | Path) -> dict[str, Any]:
    """Parse a WhiteMatter LLC binary filename to extract recording metadata.

    The filename is expected to follow the convention::

        HSW_2025_01_14__12_43_33__03min_46sec__hsamp_64ch_20000sps.bin

    Parameters
    ----------
    filename : str or Path
        Path to the WhiteMatter binary data file.

    Returns
    -------
    dict
        Header with fields: ``filename``, ``filepath``, ``file_format``,
        ``start_time_iso``, ``duration_seconds``, ``device_type``,
        ``num_channels``, ``sampling_rate``.
    """
    filename = Path(filename)
    H: dict[str, Any] = {}
    H["filename"] = filename.name
    H["filepath"] = str(filename.parent)

    name = filename.stem
    parts = name.split("__")

    if len(parts) != 4:
        raise ValueError(
            f'Filename "{filename.name}" does not match the expected format. '
            f"Expected 4 parts separated by '__', but found {len(parts)}."
        )

    # Part 1: Format and date (e.g. "HSW_2025_01_14")
    part1 = parts[0]
    first_us = part1.index("_")
    H["file_format"] = part1[:first_us]
    date_underscores = part1[first_us + 1 :]
    date_part = date_underscores.replace("_", "-")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_part):
        raise ValueError(
            f'Parsed date "{date_part}" from "{part1}" does not match YYYY-MM-DD format.'
        )

    # Part 2: Time (e.g. "12_43_33")
    time_part = parts[1].replace("_", ":")
    H["start_time_iso"] = f"{date_part}T{time_part}"

    # Part 3: Duration (e.g. "03min_46sec")
    dur_match = re.match(r"(\d+)min_(\d+)sec", parts[2])
    if not dur_match:
        raise ValueError(f'Could not parse duration "{parts[2]}".')
    H["duration_seconds"] = int(dur_match.group(1)) * 60 + int(dur_match.group(2))

    # Part 4: Device info (e.g. "hsamp_64ch_20000sps")
    dev_match = re.match(r"^([a-zA-Z0-9_]+?)_(\d+)ch_(\d+)sps$", parts[3])
    if not dev_match:
        raise ValueError(f'Could not parse device info "{parts[3]}".')
    H["device_type"] = dev_match.group(1)
    H["num_channels"] = int(dev_match.group(2))
    H["sampling_rate"] = int(dev_match.group(3))

    return H
