"""Test reading binarymatrix format.

Ported from +ndr/+test/+format/+binarymatrix/testread.m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ndr.format.binarymatrix.read import read


@pytest.fixture()
def testfile(tmp_path: Path) -> Path:
    """Create the binary test file with uint32 values 1..100."""
    filename = tmp_path / "testfile.bin"
    a = np.arange(1, 101, dtype=np.uint32)
    a.astype("<u4").tofile(filename)
    return filename


def test_10_channels_10_samples(testfile: Path) -> None:
    """Read assuming 10 channels, yielding 10 samples."""
    a = np.arange(1, 101, dtype=np.uint32)

    data_10, total_samples, s0_out, s1_out = read(
        testfile, 10, list(range(1, 11)), -np.inf, np.inf, dataType="uint32"
    )

    # Binary matrix: interleaved channels. Sample 1 = [1..10], sample 2 = [11..20], etc.
    # MATLAB: reshape(A, 10, 100/10)' -- column-major reshape then transpose
    # Python equivalent: reshape with C-order (samples x channels)
    expected_10 = a.reshape(10, 10)
    assert np.array_equal(expected_10, data_10), "10 samples of 10 channels test failed"
    assert total_samples == 10, "total_samples was not 10 in 10 samples of 10 channels test"
    assert s0_out == 1, "10 samples, 10 channels, s0_out was not 1."
    assert s1_out == 10, "10 samples, 10 channels, s1_out was not 10."


def test_10_channels_backward(testfile: Path) -> None:
    """Read 10 channels in reverse order."""
    a = np.arange(1, 101, dtype=np.uint32)

    expected_10 = a.reshape(10, 10)

    data_10bw, total_samples, s0_out, s1_out = read(
        testfile, 10, list(range(10, 0, -1)), -np.inf, np.inf, dataType="uint32"
    )
    expected_10bw = expected_10[:, ::-1]

    assert np.array_equal(expected_10bw, data_10bw), "10 samples of 10 channels backward test failed"
    assert total_samples == 10
    assert s0_out == 1
    assert s1_out == 10


def test_5_channels_20_samples(testfile: Path) -> None:
    """Read assuming 5 channels, yielding 20 samples."""
    a = np.arange(1, 101, dtype=np.uint32)

    data_5, total_samples, s0_out, s1_out = read(
        testfile, 5, list(range(1, 6)), -np.inf, np.inf, dataType="uint32"
    )

    expected_5 = a.reshape(20, 5)  # 20 samples x 5 channels
    assert np.array_equal(expected_5, data_5), "20 samples of 5 channels test failed"
    assert total_samples == 20, "total_samples was not 20 in 20 samples of 5 channels test"
    assert s0_out == 1, "20 samples, 5 channels test, s0_out was not 1."
    assert s1_out == 20, "20 samples, 5 channels test, s1_out was not 20."


def test_2_channels_50_samples(testfile: Path) -> None:
    """Read assuming 2 channels, yielding 50 samples."""
    a = np.arange(1, 101, dtype=np.uint32)

    data_2, total_samples, s0_out, s1_out = read(
        testfile, 2, list(range(1, 3)), -np.inf, np.inf, dataType="uint32"
    )

    expected_2 = a.reshape(50, 2)  # 50 samples x 2 channels
    assert np.array_equal(expected_2, data_2), "50 samples of 2 channels test failed"
    assert total_samples == 50, "total_samples was not 50 in 50 samples of 2 channels test"
    assert s0_out == 1, "50 samples, 2 channels test, s0_out was not 1."
    assert s1_out == 50, "50 samples, 2 channels test, s1_out was not 50."


def test_read_middle_samples(testfile: Path) -> None:
    """Read samples 5..7 from 10-channel arrangement."""
    a = np.arange(1, 101, dtype=np.uint32)
    expected_10 = a.reshape(10, 10)

    data_10m, total_samples, s0_out, s1_out = read(
        testfile, 10, list(range(1, 11)), 5, 7, dataType="uint32"
    )

    expected_10m = expected_10[4:7, :]  # rows 5-7 (0-indexed: 4-6)
    assert np.array_equal(expected_10m, data_10m), "samples 5..7 of 10 channels test failed"
    assert total_samples == 10, "total_samples was not 10 in 10 samples of 10 channels test"
    assert s0_out == 5, "samples 5..7, 10 channels, s0_out was not 5."
    assert s1_out == 7, "samples 5..7, 10 channels, s1_out was not 7."
