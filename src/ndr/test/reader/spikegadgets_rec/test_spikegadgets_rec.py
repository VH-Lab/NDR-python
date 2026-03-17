"""Test reading using NDR reader with SpikeGadgets .rec format.

Ported from +ndr/+test/+reader/+spikegadgets_rec/test.m and readertest.m

NOTE: This test requires example data file (example.rec) in the NDR
example_data directory. Download NDR example data to run this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ndr.globals import NDRGlobals


def _get_example_file() -> Path:
    """Return the path to the SpikeGadgets example file."""
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data" / "example.rec"


@pytest.fixture()
def example_rec() -> Path:
    """Provide path to example.rec, skip if not available."""
    f = _get_example_file()
    if not f.exists():
        pytest.skip(
            f"Example data file not found: {f}. "
            "Download NDR example data to run this test."
        )
    return f


@pytest.fixture()
def rec_reader():
    """Create a SpikeGadgets rec reader instance."""
    try:
        from ndr.reader.spikegadgets_rec import ndr_reader_spikegadgets__rec
    except ImportError:
        pytest.skip("ndr.reader.spikegadgets_rec.ndr_reader_spikegadgets__rec not yet available")
    return ndr_reader_spikegadgets__rec()


def test_reader_creation(rec_reader) -> None:
    """Test that a SpikeGadgets rec reader can be instantiated."""
    assert rec_reader is not None


@pytest.mark.xfail(reason="SpikeGadgets reader not yet fully implemented")
def test_getchannelsepoch(rec_reader, example_rec: Path) -> None:
    """Test getting channel information from a SpikeGadgets rec epoch."""
    # The MATLAB test calls:
    #   r = ndr.reader('rec')
    #   channels = r.getchannelsepoch({filename})
    channels = rec_reader.getchannelsepoch([str(example_rec)])

    assert channels is not None
    assert len(channels) > 0

    for ch in channels:
        assert hasattr(ch, "name") or "name" in ch
        assert hasattr(ch, "type") or "type" in ch


@pytest.mark.xfail(reason="SpikeGadgets reader not yet fully implemented")
def test_readchannels_epochsamples(rec_reader, example_rec: Path) -> None:
    """Test reading epoch samples from a SpikeGadgets rec file (channel 120)."""
    # The MATLAB test calls:
    #   data = r.readchannels_epochsamples('analog_in', 120, {filename}, 1, 1, 10000)
    #   time = r.readchannels_epochsamples('time', 120, {filename}, 1, 1, 10000)
    epoch_select = 1
    channel = 120

    data = rec_reader.readchannels_epochsamples(
        "analog_in", channel, [str(example_rec)], epoch_select, 1, 10000
    )
    time = rec_reader.readchannels_epochsamples(
        "time", channel, [str(example_rec)], epoch_select, 1, 10000
    )

    assert data is not None
    assert time is not None
    assert len(data) == len(time)


def test_epochclock(rec_reader, example_rec: Path) -> None:
    """Test getting epoch clock and t0/t1 from a SpikeGadgets rec file."""
    epoch_select = 1

    ec = rec_reader.epochclock([str(example_rec)], epoch_select)
    t0t1 = rec_reader.t0_t1([str(example_rec)], epoch_select)

    assert ec is not None
    assert t0t1 is not None


@pytest.mark.xfail(reason="SpikeGadgets reader not yet fully implemented")
def test_readertest(rec_reader, example_rec: Path) -> None:
    """Test the reader.read convenience function (from readertest.m).

    Reads analog input channel 120.
    """
    d, t = rec_reader.read([str(example_rec)], "ai120")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
