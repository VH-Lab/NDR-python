"""NDR path utilities.

Port of +ndr/+fun/ndrpath.m
"""

from __future__ import annotations

from pathlib import Path


def ndrpath() -> Path:
    """Return the root path of the NDR-python package.

    Returns
    -------
    Path
        The root directory of the NDR package (src/ndr).
    """
    return Path(__file__).parent.parent
