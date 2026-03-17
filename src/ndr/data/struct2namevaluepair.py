"""Convert a dict to a flat list of name/value pairs.

Port of +ndr/+data/struct2namevaluepair.m
"""

from __future__ import annotations

from typing import Any


def struct2namevaluepair(thestruct: dict[str, Any]) -> list[Any]:
    """Convert a dictionary to a flat list of name/value pairs.

    This is useful for passing name/value pairs to functions that accept
    them as extra keyword arguments. Each key of the dictionary is used as
    the 'name', and the corresponding value is used as the 'value'.

    Parameters
    ----------
    thestruct : dict
        Input dictionary mapping parameter names to values.

    Returns
    -------
    list
        Flat list alternating between keys and values,
        e.g. ``['param1', 1, 'param2', 2]``.

    Examples
    --------
    >>> struct2namevaluepair({'param1': 1, 'param2': 2})
    ['param1', 1, 'param2', 2]
    """
    if not thestruct:
        return []

    nv: list[Any] = []
    for key, value in thestruct.items():
        nv.append(key)
        nv.append(value)
    return nv
