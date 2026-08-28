"""Read a Prairie View XML key/value parameter.

Port of +ndr/+format/+prairieview/keyvalue.m
"""

from __future__ import annotations

import re


def keyvalue(txt: str, keyname: str) -> float | str | None:
    """Return the value of a ``<Key key="KEYNAME" value="...">`` element.

    Given the text of a modern (PVScan) Prairie View XML file, returns the
    value of the ``<Key key="KEYNAME" value="...">`` or
    ``<PVStateValue key="KEYNAME" value="...">`` element. The value is
    returned as a number when it parses as numeric, otherwise as the raw
    string. Returns ``None`` if the key is not found.

    This is a revised port of ``readprairie3keyvalue.m`` from
    VH-Lab/vhlab-TwoPhoton-matlab (Platforms/PrairieView).
    """
    # Match a 'key="<keyname>" ... value="..."' pair within a single tag (the
    # [^>]* cannot cross a '>'), so key and value belong to the same
    # <Key>/<PVStateValue> element.
    pat = r'key="' + re.escape(keyname) + r'"[^>]*value="([^"]*)"'
    m = re.search(pat, txt)
    if m is None:
        return None

    raw = m.group(1)
    try:
        return float(raw)
    except ValueError:
        return raw
