"""Data type string utility.

Port of +ndr/+fun/getDataTypeString.m
"""

from __future__ import annotations


def getDataTypeString(bits: int, is_float: bool = False, is_signed: bool = True) -> str:
    """Return a data type string for a given bit depth.

    Parameters
    ----------
    bits : int
        Number of bits.
    is_float : bool
        Whether the type is floating point.
    is_signed : bool
        Whether the type is signed (for integers).

    Returns
    -------
    str
        Data type string suitable for numpy (e.g., 'float64', 'uint16').
    """
    if is_float:
        return f"float{bits}"
    prefix = "int" if is_signed else "uint"
    return f"{prefix}{bits}"
