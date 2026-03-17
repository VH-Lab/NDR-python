"""Test reading using NDR reader with Intan Technologies .RHD file format.

Ported from +ndr/+test/+reader/+intan_rhd/test.m and readertest.m

NOTE: This test requires example data file (example.rhd) in the NDR
example_data directory. Download NDR example data to run this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ndr.globals import NDRGlobals


def _get_example_file() -> Path:
    """Return the path to the Intan RHD example file."""
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data" / "example.rhd"


@pytest.fixture()
def example_rhd() -> Path:
    """Provide path to example.rhd, skip if not available."""
    f = _get_example_file()
    if not f.exists():
        pytest.skip(
            f"Example data file not found: {f}. "
            "Download NDR example data to run this test."
        )
    return f


@pytest.fixture()
def intan_reader():
    """Create an Intan RHD reader instance."""
    try:
        from ndr.reader.intan_rhd import ndr_reader_intan__rhd
    except ImportError:
        pytest.skip("ndr.reader.intan_rhd.ndr_reader_intan__rhd not yet available")
    return ndr_reader_intan__rhd()


def test_reader_creation(intan_reader) -> None:
    """Test that an Intan RHD reader can be instantiated."""
    assert intan_reader is not None


def test_getchannelsepoch(intan_reader, example_rhd: Path) -> None:
    """Test getting channel information from an Intan RHD epoch."""
    # The MATLAB test calls:
    #   r = ndr.reader('intan')
    #   channels = r.getchannelsepoch({filename})
    channels = intan_reader.getchannelsepoch([str(example_rhd)])

    assert channels is not None
    assert len(channels) > 0

    for ch in channels:
        assert hasattr(ch, "name") or "name" in ch
        assert hasattr(ch, "type") or "type" in ch


def test_readchannels_epochsamples(intan_reader, example_rhd: Path) -> None:
    """Test reading epoch samples from an Intan RHD file (channel 32)."""
    # The MATLAB test calls:
    #   d = r.readchannels_epochsamples('ai', 32, {filename}, 1, 1, 10000)
    #   t = r.readchannels_epochsamples('time', 32, {filename}, 1, 1, 10000)
    epoch_select = 1
    channel = 32

    d = intan_reader.readchannels_epochsamples(
        "ai", channel, [str(example_rhd)], epoch_select, 1, 10000
    )
    t = intan_reader.readchannels_epochsamples(
        "time", channel, [str(example_rhd)], epoch_select, 1, 10000
    )

    assert d is not None
    assert t is not None
    assert len(d) == len(t)


def test_epochclock(intan_reader, example_rhd: Path) -> None:
    """Test getting epoch clock and t0/t1 from an Intan RHD file."""
    epoch_select = 1

    ec = intan_reader.epochclock([str(example_rhd)], epoch_select)
    t0t1 = intan_reader.t0_t1([str(example_rhd)], epoch_select)

    assert ec is not None
    assert t0t1 is not None


def test_readertest(intan_reader, example_rhd: Path) -> None:
    """Test the reader.read convenience function (from readertest.m).

    Reads channel A021.
    """
    d, t = intan_reader.read([str(example_rhd)], "A021")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
