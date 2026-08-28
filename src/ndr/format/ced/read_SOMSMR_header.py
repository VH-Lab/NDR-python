"""Read a CED Spike2 file header.

Port of +ndr/+format/+ced/read_SOMSMR_header.m

All CED reads in this port go through the sonpipe CLI, which drives CED's own
sonpy binding. NDR-matlab routes 32-bit .smr files to sigTOOL and only 64-bit
.smrx files to sonpipe; that split exists because sigTOOL has years of
production use behind it, and it is deliberately not mirrored here. Python's
alternative was neo's CedRawIO, which is a reimplementation rather than a
battle-tested reader, and which depends on sonpy being importable in-process --
impossible on Linux and macOS for CPython 3.10-3.13, and on Apple Silicon for
any version. One backend, from the vendor, beats two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ndr.format.ced.sonpipe.read_SOMSMR_header import (
    read_SOMSMR_header as _sonpipe_read_SOMSMR_header,
)


def read_SOMSMR_header(filename: str | Path) -> dict[str, Any]:
    """Return the file header as ``{"fileinfo": ..., "channelinfo": [...]}``."""
    return _sonpipe_read_SOMSMR_header(filename)
