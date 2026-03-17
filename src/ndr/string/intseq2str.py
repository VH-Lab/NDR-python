"""Integer sequence to string conversion.

Port of +ndr/+string/intseq2str.m
"""

from __future__ import annotations


def intseq2str(seq: list[int]) -> str:
    """Convert a list of integers to a compact range string.

    Parameters
    ----------
    seq : list of int
        Integer sequence.

    Returns
    -------
    str
        Compact string representation (e.g., [1,2,3,5] -> '1-3,5').
    """
    if not seq:
        return ""

    sorted_seq = sorted(set(seq))
    parts: list[str] = []
    i = 0

    while i < len(sorted_seq):
        start = sorted_seq[i]
        end = start
        while i + 1 < len(sorted_seq) and sorted_seq[i + 1] == end + 1:
            end = sorted_seq[i + 1]
            i += 1
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
        i += 1

    return ",".join(parts)
