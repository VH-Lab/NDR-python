"""Read Qt-style QString from a binary file.

Port of +ndr/+format/+intan/fread_QString.m
"""

from __future__ import annotations

import struct


def fread_QString(f) -> str:
    """Read a Qt-style QString from a binary file.

    The first 32-bit unsigned number indicates the length of the string
    (in bytes).  If this number equals 0xFFFFFFFF, the string is null.

    Parameters
    ----------
    f : file-like
        An open binary file object positioned at the start of a QString.

    Returns
    -------
    str
        The decoded string, or empty string if null.
    """
    length_bytes = f.read(4)
    if len(length_bytes) < 4:
        return ""
    length = struct.unpack("<I", length_bytes)[0]
    if length == 0xFFFFFFFF:
        return ""
    data = f.read(length)
    return data.decode("utf-16-le", errors="replace")
