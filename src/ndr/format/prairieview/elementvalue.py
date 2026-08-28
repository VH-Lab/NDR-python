"""Read an element value from a (legacy) Prairie View XML.

Port of +ndr/+format/+prairieview/elementvalue.m
"""

from __future__ import annotations

import re


def elementvalue(txt: str, tag: str) -> float | str | None:
    """Return the value enclosed by the first element whose tag contains ``tag``.

    Given the text of a legacy (MM-era) Prairie View XML file, returns the text
    between ``'>'`` and the next ``'<'`` of ``<...TAG...>VALUE<...>``. The value
    is returned as a number when it parses as numeric, otherwise as the raw
    string. Returns ``None`` if the tag is not found.

    This is a revised port of the ``getxmlval`` subfunction of
    ``readprairieviewxml.m`` from VH-Lab/vhlab-TwoPhoton-matlab
    (Platforms/PrairieView).
    """
    pat = r"<[^>]*" + re.escape(tag) + r"[^>]*>([^<]*)<"
    m = re.search(pat, txt)
    if m is None:
        return None

    raw = m.group(1)
    try:
        return float(raw)
    except ValueError:
        return raw
