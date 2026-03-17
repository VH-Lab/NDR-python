"""String to integer sequence parsing.

Port of +ndr/+string/str2intseq.m
"""

from __future__ import annotations


def str2intseq(s: str) -> list[int]:
    """Convert a string with comma-separated numbers and dash ranges to a list of ints.

    Parameters
    ----------
    s : str
        Number specification (e.g., '1,3-5,2' -> [1, 3, 4, 5, 2]).

    Returns
    -------
    list of int
        The parsed integer sequence.
    """
    s = s.strip()
    if not s:
        return []

    result: list[int] = []
    parts = s.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like '3-5'
            range_parts = part.split("-", 1)
            start = int(range_parts[0].strip())
            end = int(range_parts[1].strip())
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    return result
