"""Test reading using NDR reader with TDT SEV format.

Ported from +ndr/+test/+reader/+tdt_sev/test.m and readertest.m

NOTE: This test requires a TDT SEV example data file. The original MATLAB
test expects a filename argument. Download NDR example data to run this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ndr.globals import NDRGlobals


def _get_example_dir() -> Path:
    """Return the path to the NDR example_data directory."""
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data"


@pytest.fixture()
def example_sev() -> Path:
    """Provide path to an example SEV file, skip if not available."""
    example_dir = _get_example_dir()
    sev_files = list(example_dir.glob("*.sev"))
    if not sev_files:
        pytest.skip(
            f"No example SEV files found in {example_dir}. "
            "Download NDR example data to run this test."
        )
    return sev_files[0]


@pytest.fixture()
def sev_reader():
    """Create a TDT SEV reader instance."""
    try:
        from ndr.reader.tdt_sev import ndr_reader_tdt__sev
    except ImportError:
        pytest.skip("ndr.reader.tdt_sev.ndr_reader_tdt__sev not yet available")
    return ndr_reader_tdt__sev()


def test_reader_creation(sev_reader) -> None:
    """Test that a TDT SEV reader can be instantiated."""
    assert sev_reader is not None


def test_getchannelsepoch(sev_reader, example_sev: Path) -> None:
    """Test getting channel information from a TDT SEV epoch."""
    # The MATLAB test calls:
    #   r = ndr.reader('sev')
    #   channels = r.getchannelsepoch({filename})
    channels = sev_reader.getchannelsepoch([str(example_sev)])

    assert channels is not None
    assert len(channels) > 0

    for ch in channels:
        assert hasattr(ch, "name") or "name" in ch
        assert hasattr(ch, "type") or "type" in ch


def test_readchannels_epochsamples(sev_reader, example_sev: Path) -> None:
    """Test reading epoch samples from a TDT SEV file (channel 32)."""
    # The MATLAB test calls:
    #   d = r.readchannels_epochsamples('ai', 32, {filename}, 1, 1, 10000)
    #   t = r.readchannels_epochsamples('time', 32, {filename}, 1, 1, 10000)
    epoch_select = 1
    channel = 32

    d = sev_reader.readchannels_epochsamples(
        "ai", channel, [str(example_sev)], epoch_select, 1, 10000
    )
    t = sev_reader.readchannels_epochsamples(
        "time", channel, [str(example_sev)], epoch_select, 1, 10000
    )

    assert d is not None
    assert t is not None
    assert len(d) == len(t)


def test_epochclock(sev_reader, example_sev: Path) -> None:
    """Test getting epoch clock and t0/t1 from a TDT SEV file."""
    epoch_select = 1

    ec = sev_reader.epochclock([str(example_sev)], epoch_select)
    t0t1 = sev_reader.t0_t1([str(example_sev)], epoch_select)

    assert ec is not None
    assert t0t1 is not None


def test_readertest(sev_reader, example_sev: Path) -> None:
    """Test the reader.read convenience function (from readertest.m).

    Reads analog input channel 1.
    """
    d, t = sev_reader.read([str(example_sev)], "ai1")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
