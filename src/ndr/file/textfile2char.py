"""Text file reading utility.

Port of +ndr/+file/textfile2char.m
"""

from __future__ import annotations

from pathlib import Path


def textfile2char(filepath: str | Path) -> str:
    """Read a text file and return its contents as a string.

    Parameters
    ----------
    filepath : str or Path
        Path to the text file.

    Returns
    -------
    str
        File contents.
    """
    return Path(filepath).read_text(encoding="utf-8")
