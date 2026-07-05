"""Read an element value from a (legacy) Prairie View XML.

Port of +ndr/+format/+prairieview/elementvalue.m
"""

from __future__ import annotations

import re
from typing import Union

from ndr.format.prairieview.keyvalue import _str2double


def elementvalue(txt: str, tag: str) -> Union[float, str, None]:
    """Read an element value from a legacy (MM-era) Prairie View XML.

    Given the text ``txt`` of a legacy Prairie View XML file, return the value
    enclosed by the first element whose tag contains ``tag``, i.e. the text
    between ``>`` and the next ``<`` of ``<...TAG...>VALUE<...>``. The value is
    returned as a number when it parses as numeric, otherwise as the raw
    string. Returns ``None`` if the tag is not found.

    This is a faithful port of ``ndr.format.prairieview.elementvalue`` (itself a
    revised port of the ``getxmlval`` subfunction of ``readprairieviewxml.m``
    from VH-Lab/vhlab-TwoPhoton-matlab).

    Parameters
    ----------
    txt : str
        The full text of the legacy Prairie View XML file.
    tag : str
        A substring that must appear inside the element's opening tag.

    Returns
    -------
    float or str or None
        The value as a number if it parses as numeric; otherwise the raw
        string; ``None`` if the tag is not present.
    """
    pat = "<[^>]*" + re.escape(tag) + "[^>]*>([^<]*)<"
    m = re.search(pat, txt)
    if m is None:
        return None
    raw = m.group(1)
    val = _str2double(raw)
    if val is None:
        return raw
    return val
