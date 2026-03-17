"""Assign name/value pairs into a target namespace (dict).

Port of +ndr/+data/assign.m
"""

from __future__ import annotations

from typing import Any

from ndr.data.struct2namevaluepair import struct2namevaluepair


def assign(target: dict[str, Any], *args: Any) -> dict[str, Any]:
    """Apply a list of name/value pair assignments to *target*.

    In MATLAB ``ndr.data.assign`` uses ``assignin('caller', ...)`` to inject
    variables into the caller's workspace.  In Python the idiomatic
    equivalent is to update a dictionary (typically ``locals()`` or an
    options dict) and return it.

    Parameters
    ----------
    target : dict
        The dictionary to update with the supplied name/value pairs.
    *args
        Either a single ``dict`` (struct equivalent), a single ``list``
        of alternating name/value items, or inline alternating
        ``name, value, name, value, ...`` arguments.

    Returns
    -------
    dict
        The updated *target* dictionary (same object, mutated in-place).

    Examples
    --------
    >>> opts = {'z': 0}
    >>> assign(opts, 'z', 4)
    {'z': 4}

    >>> assign({}, {'a': 1, 'b': 2})
    {'a': 1, 'b': 2}

    >>> assign({}, ['x', 10, 'y', 20])
    {'x': 10, 'y': 20}
    """
    # Normalise a single-argument form (dict or list) into a flat sequence
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            flat: list[Any] = struct2namevaluepair(arg)
        elif isinstance(arg, (list, tuple)):
            flat = list(arg)
        else:
            raise TypeError(
                "A single argument must be a dict or a list of name/value pairs."
            )
    else:
        flat = list(args)

    names = flat[0::2]
    values = flat[1::2]

    for name, value in zip(names, values):
        target[name] = value

    return target
