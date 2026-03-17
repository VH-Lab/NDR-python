"""Bit depth utility.

Port of +ndr/+fun/bitDepth.m
"""

from __future__ import annotations


def bitDepth(datatype: str) -> int:
    """Return the bit depth for a given data type string.

    Parameters
    ----------
    datatype : str
        Data type string (e.g., 'uint16', 'float64').

    Returns
    -------
    int
        Number of bits.
    """
    bit_map = {
        "int8": 8,
        "uint8": 8,
        "char": 8,
        "int16": 16,
        "uint16": 16,
        "int32": 32,
        "uint32": 32,
        "float32": 32,
        "single": 32,
        "int64": 64,
        "uint64": 64,
        "float64": 64,
        "double": 64,
    }
    if datatype not in bit_map:
        raise ValueError(f"Unknown data type '{datatype}'.")
    return bit_map[datatype]
