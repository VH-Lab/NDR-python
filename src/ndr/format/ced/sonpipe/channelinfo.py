"""Select one channel's metadata from a sonpipe header.

Port of +ndr/+format/+ced/+sonpipe/channelinfo.m
"""

from __future__ import annotations

from typing import Any


def channelinfo(header: dict[str, Any] | None, channel_number: int) -> dict[str, Any]:
    """Return the entry in ``header["channelinfo"]`` whose number matches."""
    if not header or not header.get("channelinfo"):
        raise ValueError("The header contains no channels.")

    for entry in header["channelinfo"]:
        if int(entry["number"]) == int(channel_number):
            return entry

    raise ValueError(f"Channel number {channel_number} is not recorded in this file.")
