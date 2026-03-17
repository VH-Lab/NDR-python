"""Channel string parsing utilities.

Port of +ndr/+string/channelstring2channels.m
"""

from __future__ import annotations

from ndr.string.str2intseq import str2intseq


def channelstring2channels(channelstring: str) -> tuple[list[str], list[int]]:
    """Convert a channel string to arrays of channel prefixes and numbers.

    A channel string specifies channels with prefixes and numbers.
    Use '-' for sequential ranges, ',' to enumerate, and '+' to separate
    different channel prefix groups.

    Parameters
    ----------
    channelstring : str
        Channel specification string (e.g., 'ai1-3', 'ai1,3-5+di2-4').

    Returns
    -------
    tuple of (list[str], list[int])
        (channel_name_prefixes, channel_numbers)

    Examples
    --------
    >>> channelstring2channels('a1,3-5,2')
    (['a', 'a', 'a', 'a', 'a'], [1, 3, 4, 5, 2])
    >>> channelstring2channels('ai1-3+b2-4')
    (['ai', 'ai', 'ai', 'b', 'b', 'b'], [1, 2, 3, 2, 3, 4])
    """
    channelstring = channelstring.strip()
    block_separator = "+"

    blocks = channelstring.split(block_separator)

    channelnameprefix: list[str] = []
    channelnumber: list[int] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Find the first non-letter character
        first_non_letter = None
        for i, ch in enumerate(block):
            if not ch.isalpha():
                first_non_letter = i
                break

        if first_non_letter is None:
            raise ValueError(f"No numbers provided in channel string segment '{block}'.")

        prefix = block[:first_non_letter]
        numbers = str2intseq(block[first_non_letter:])

        channelnameprefix.extend([prefix] * len(numbers))
        channelnumber.extend(numbers)

    return channelnameprefix, channelnumber
