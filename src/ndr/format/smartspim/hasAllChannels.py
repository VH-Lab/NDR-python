"""Non-throwing probe for a usable SmartSPIM ``All_Channels`` volume.

Port of ``+ndr/+format/+smartspim/hasAllChannels.m``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ndr.format.smartspim.readAllChannelsInfo import readAllChannelsInfo

__all__ = ["hasAllChannels"]


def hasAllChannels(rootDir: Any) -> bool:
    """Return True if ``rootDir/All_Channels`` is usable.

    True when ``All_Channels`` holds SmartSPIM's stitched,
    flat-field-corrected slices for EVERY channel of the acquisition;
    False otherwise (folder missing, no slices, a channel's ``_ch<k>``
    slices missing, unreadable or non-path input).

    Only file names are inspected; no TIFF header is opened. Readers use
    this to prefer the stitched ``All_Channels`` volume and to fall back
    to the per-tile ``Ex_*`` layout for acquisitions where AutoStitch has
    not been run (or ``All_Channels`` was not restored from the archive).
    Takes any input (like the MATLAB probe) so it never raises.
    """
    if not isinstance(rootDir, (str, Path)) or not str(rootDir):
        return False
    if not (Path(rootDir) / "All_Channels").is_dir():
        return False
    try:
        readAllChannelsInfo(str(rootDir), ReadHeader=False)
    except Exception:
        return False
    return True
