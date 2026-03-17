"""Test creating a binary matrix test file.

Ported from +ndr/+test/+format/+binarymatrix/maketestfile.m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def maketestfile(filepath: Path | None = None) -> Path:
    """Create a binary file with uint32 numbers from 1..100 in little-endian format.

    Parameters
    ----------
    filepath : Path or None
        Directory in which to place testfile.bin.  Defaults to the same
        directory as this Python file.

    Returns
    -------
    Path
        The path to the created file.
    """
    if filepath is None:
        filepath = Path(__file__).parent

    filename = filepath / "testfile.bin"

    a = np.arange(1, 101, dtype=np.uint32)

    with open(filename, "wb") as f:
        # Write using little-endian byte order
        a.astype("<u4").tofile(f)

    return filename


def test_maketestfile(tmp_path: Path) -> None:
    """Verify maketestfile creates a valid binary file."""
    filename = maketestfile(tmp_path)
    assert filename.exists()

    data = np.fromfile(filename, dtype="<u4")
    expected = np.arange(1, 101, dtype=np.uint32)

    assert data.shape == (100,)
    assert np.array_equal(data, expected)
