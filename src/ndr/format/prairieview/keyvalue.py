"""Read a Prairie View XML key/value parameter.

Port of +ndr/+format/+prairieview/keyvalue.m
"""

from __future__ import annotations

import re
from typing import Union


def keyvalue(txt: str, keyname: str) -> Union[float, str, None]:
    """Read a Prairie View (modern PVScan) XML key/value parameter.

    Given the text ``txt`` of a modern (PVScan) Prairie View XML file, return
    the value of the ``<Key key="KEYNAME" value="...">`` or
    ``<PVStateValue key="KEYNAME" value="...">`` element. The value is returned
    as a ``float`` when it parses as numeric, otherwise as the raw string.
    Returns ``None`` if the key is not found.

    This is a faithful port of ``ndr.format.prairieview.keyvalue`` (itself a
    revised port of ``readprairie3keyvalue.m`` from
    VH-Lab/vhlab-TwoPhoton-matlab): the same key/value tag model, parsed with a
    regular expression.

    Parameters
    ----------
    txt : str
        The full text of the (modern PVScan) Prairie View XML file.
    keyname : str
        The ``key`` attribute to look up.

    Returns
    -------
    float or str or None
        The value as a number if it parses as numeric; otherwise the raw
        string; ``None`` if the key is not present.
    """
    # Match a 'key="<keyname>" ... value="..."' pair within a single tag
    # (the [^>]* cannot cross a '>'), so key and value belong to the same
    # <Key>/<PVStateValue> element.
    pat = 'key="' + re.escape(keyname) + '"[^>]*value="([^"]*)"'
    m = re.search(pat, txt)
    if m is None:
        return None
    raw = m.group(1)
    val = _str2double(raw)
    if val is None:  # not a plain number; keep the string
        return raw
    return val


def _str2double(s: str) -> Union[float, None]:
    """Mimic MATLAB ``str2double``: parse a scalar number or return ``None``.

    Returns ``None`` when the (trimmed) string does not parse as a single real
    number, which is how MATLAB signals ``NaN`` for a non-numeric string here.
    """
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None
