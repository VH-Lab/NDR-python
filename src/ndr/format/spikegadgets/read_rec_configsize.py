"""Locate the first byte of packet data in a SpikeGadgets .rec file.

A .rec begins with an XML configuration block terminated by
``</Configuration>``, and the packet stream starts on the NEXT LINE.

This exists to fix an index-base slip in the port. NDR-matlab computes the
offset as ``strfind(junk, '</Configuration>') + 16``, and strfind is 1-BASED,
so that lands on the first packet byte -- MATLAB has always been right. The
Python port copied the ``+ 16`` onto ``bytes.find``, which is 0-BASED, so it
landed one byte earlier, on the line terminator. Every subsequent read was
shifted by a byte, assembling each int16 from the high byte of one sample and
the low byte of the next: plausible-looking values that no size or range check
can catch.

Rather than hardcoding a compensating ``+ 17``, this consumes the terminator
explicitly, which is what Trodes' reference reader and neo's
SpikeGadgetsRawIO do (the latter reads the header line by line and takes
``f.tell()`` after the line containing the tag). For the single newline Trodes
writes, this agrees with MATLAB exactly. It also stays correct for a CRLF
terminator or a file written without one, where MATLAB's fixed offset would
be wrong -- a divergence in Python's favour, noted in the bridge YAML.
"""

from __future__ import annotations

from pathlib import Path

_TAG = b"</Configuration>"


def read_rec_configsize(filename: str | Path, search_bytes: int = 1_000_000) -> int:
    """Return the byte offset where packet data begins.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.rec`` file.
    search_bytes : int
        How much of the file head to search for the terminating tag.

    Returns
    -------
    int
        Offset of the first packet byte, or 0 if the file has no configuration
        block (some raw SD-card captures do not).
    """
    with open(filename, "rb") as f:
        head = f.read(search_bytes)

    index = head.find(_TAG)
    if index < 0:
        return 0

    offset = index + len(_TAG)
    # Consume the line terminator, so the offset lands on the first packet byte
    # rather than on the newline. Guarded rather than assumed: a file written
    # without a trailing newline stays correct.
    if head[offset : offset + 2] == b"\r\n":
        return offset + 2
    if head[offset : offset + 1] in (b"\n", b"\r"):
        return offset + 1
    return offset
