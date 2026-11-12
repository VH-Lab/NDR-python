"""SmartSPIM's record of which channel folder is which ``Ch<k>``.

Port of ``+ndr/+format/+smartspim/readImsNotes.m``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = ["readImsNotes"]

_LINE_RE = re.compile(r"The folder\s+(\S+?)_stitched\s+is represented as\s+Ch(\d+)", re.IGNORECASE)


def readImsNotes(rootDir: Any) -> list[dict[str, Any]]:
    """Read ``rootDir/IMS_Notes.txt``: which channel folder is which ``Ch<k>``.

    The SmartSPIM software writes it when it converts the stitched
    ``All_Channels`` volume to Imaris, one line per channel, e.g.::

        The folder Ex_561_Em_561F_stitched is represented as Ch0 with a Hex color value of #ff0000

    Returns one dict per such line: ``channelName`` (the channel folder, the
    named ``<channel>_stitched`` folder without ``_stitched``) and
    ``chIndex`` (float, the ``k`` of that channel's ``All_Channels``
    ``_Ch<k>`` slices). This is the stitcher's own statement of the mapping,
    so :func:`readAllChannelsInfo` prefers it to inferring the mapping from
    filter-channel numbers. An absent or unreadable file, or one with no such
    lines, gives ``[]``; it never raises.
    """
    if not isinstance(rootDir, (str, Path)) or not str(rootDir):
        return []
    p = Path(rootDir) / "IMS_Notes.txt"
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return []
    return [
        {"channelName": m.group(1), "chIndex": float(m.group(2))} for m in _LINE_RE.finditer(text)
    ]
