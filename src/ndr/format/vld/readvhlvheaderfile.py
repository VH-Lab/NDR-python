"""Read a VHLAB LabView (VHLV) ``.vlh`` header file.

Port of +ndr/+format/+vld/readvhlvheaderfile.m

The header is a small text file where each meaningful line is a field name
followed by a colon+tab (``:\\t``) separator and then the field value. The
MATLAB reader attempts to coerce each value to a number (via ``eval``) and
falls back to a string when that fails; this port mirrors that behaviour by
trying ``int`` then ``float`` and finally keeping the raw string.

Expected fields (see the MATLAB reference):

    ChannelString   : names of the channels acquired in LabView
    NumChans        : number of channels acquired
    SamplingRate    : per-channel sampling rate in Hz
    SamplesPerChunk : number of samples written per burst
    Multiplexed     : 1 if adjacent samples are from different channels, else 0
    Scale           : voltage scale factor (optional)
    precision       : stored numeric precision (e.g. 'int16'; optional)
"""

from __future__ import annotations

import os
from typing import Any


def readvhlvheaderfile(myfilename: str) -> dict[str, Any]:
    """Parse a VHLV ``.vlh`` header file into a dict.

    Parameters
    ----------
    myfilename : str
        Path to the ``.vlh`` text header file.

    Returns
    -------
    dict
        Mapping of header field name to value. Numeric-looking values are
        coerced to ``int`` (when integral) or ``float``; everything else is
        kept as a stripped ``str``. Keys are preserved verbatim
        (e.g. ``'NumChans'``, ``'SamplingRate'``, ``'Scale'``, ``'precision'``).

    Raises
    ------
    ValueError
        If a ``.vld`` data file is passed by mistake, or the file cannot be
        opened.
    """
    _, ext = os.path.splitext(myfilename)
    if ext.lower() == ".vld":
        raise ValueError(
            f"It appears you are trying to open a data file {myfilename} "
            "with the code that reads the header."
        )

    try:
        with open(myfilename, encoding="latin-1") as fid:
            text = fid.read()
    except OSError as err:
        raise ValueError(f"Could not open file {myfilename}.") from err

    header: dict[str, Any] = {}
    # Mirror the MATLAB parser: a line contributes a field only if it contains
    # the ':\t' separator; the field name is everything before it, the value is
    # everything after it (on that line).
    for line in text.replace("\r", "").split("\n"):
        sep = line.find(":\t")
        if sep < 0:
            continue
        field_name = line[:sep]
        field_value = line[sep + 2 :]
        header[field_name] = _coerce(field_value)

    return header


def _coerce(value_string: str) -> Any:
    """Coerce a header value string to int/float when possible, else str.

    Mirrors the MATLAB ``try eval(...) catch string`` behaviour: integral
    numbers become ``int``, other numbers become ``float``, and non-numeric
    text is returned as a stripped ``str``.
    """
    stripped = value_string.strip()
    try:
        as_float = float(stripped)
    except ValueError:
        return stripped
    if as_float.is_integer() and stripped.lstrip("+-").isdigit():
        return int(as_float)
    return as_float
