"""Regression test: read_abf restores the caller's requested channel order.

read_abf sorts channel_numbers before reading (mirroring MATLAB) but previously
omitted the un-sort step, so a request like [2, 1] came back with columns
[1, 2] -- a silent column swap. This test reads a channel pair in both orders
and asserts the columns swap accordingly.

The shipped example.abf currently has a single analog channel, so the
order-restoration assertion is skipped pending a >=2-channel fixture; the
single-channel path is still exercised as a smoke check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ndr.globals import NDRGlobals

pyabf = pytest.importorskip("pyabf")

from ndr.format.axon.read_abf import read_abf  # noqa: E402


def _example_abf() -> Path:
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data" / "example.abf"


@pytest.fixture()
def example_abf() -> Path:
    f = _example_abf()
    if not f.exists():
        pytest.skip(f"Example data file not found: {f}.")
    return f


def _analog_channel_count(filename: Path) -> int:
    return int(pyabf.ABF(str(filename)).channelCount)


def test_channel_order_is_restored(example_abf: Path):
    n = _analog_channel_count(example_abf)
    if n < 2:
        pytest.skip(
            f"example.abf has {n} analog channel(s); the channel-order "
            "regression needs a >=2-channel fixture (test deferred)."
        )

    data_12 = read_abf(str(example_abf), None, "ai", [1, 2], 0.0, 0.1)
    data_21 = read_abf(str(example_abf), None, "ai", [2, 1], 0.0, 0.1)

    assert data_12.shape == data_21.shape
    # Column 0 of the [2,1] read must equal column 1 of the [1,2] read, etc.
    assert np.array_equal(data_21[:, 0], data_12[:, 1])
    assert np.array_equal(data_21[:, 1], data_12[:, 0])


def test_single_channel_read_smoke(example_abf: Path):
    data = read_abf(str(example_abf), None, "ai", [1], 0.0, 0.1)
    assert data.ndim == 2
    assert data.shape[1] == 1
    assert data.shape[0] > 0
