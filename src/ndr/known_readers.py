"""Known NDR reader types.

Port of +ndr/known_readers.m
"""

from __future__ import annotations

from ndr.fun.ndrresource import ndrresource


def known_readers() -> list[list[str]]:
    """Return all known reader file types for NDR readers.

    Returns
    -------
    list of list of str
        Each entry is the list of type aliases for one reader.
    """
    j = ndrresource("ndr_reader_types.json")
    return [entry["type"] for entry in j]
