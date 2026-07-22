"""Regression test: the CED SMR 'time' channel resolves per requested channel.

readchannels_epochsamples for channeltype 'time' used to ignore the channel
argument and always return analogsignals[0].times as a single column. It now
returns one column per requested channel, each with that channel's own time
base. On a multi-rate file this would also expose a wrong (compressed/stretched)
time axis; the shipped example.smr is single-rate, so that discrimination is
deferred, but the per-channel resolution and column count are still asserted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ndr.globals import NDRGlobals

pytest.importorskip("neo")


def _example_smr() -> Path:
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data" / "example.smr"


@pytest.fixture()
def reader_and_file():
    f = _example_smr()
    if not f.exists():
        pytest.skip(f"Example data file not found: {f}.")
    from ndr.reader.ced_smr import ndr_reader_ced__smr

    return ndr_reader_ced__smr(), [str(f)]


def _analog_channel_numbers(reader, files) -> list[int]:
    nums = []
    for c in reader.getchannelsepoch(files):
        if c.get("type") in ("analog_in", "ai"):
            digits = "".join(ch for ch in c.get("name", "") if ch.isdigit())
            if digits:
                nums.append(int(digits))
    return nums


def test_time_returns_one_column_per_channel(reader_and_file):
    reader, files = reader_and_file
    nums = _analog_channel_numbers(reader, files)
    if len(nums) < 2:
        pytest.skip("Need >=2 analog channels to check multi-column time output.")

    pair = nums[:2]
    t = reader.readchannels_epochsamples("time", pair, files, 1, 1, 10)
    # Old behavior returned exactly one column regardless of the request.
    assert t.shape[1] == len(pair)


def test_time_spacing_matches_channel_samplerate(reader_and_file):
    reader, files = reader_and_file
    nums = _analog_channel_numbers(reader, files)
    if not nums:
        pytest.skip("No analog channels in example.smr.")

    for num in nums:
        sr = reader.samplerate(files, 1, "ai", num)
        t = reader.readchannels_epochsamples("time", [num], files, 1, 1, 20)
        assert t.shape[1] == 1
        dt = np.diff(t.flatten())
        assert np.allclose(dt, 1.0 / sr, rtol=1e-6)
