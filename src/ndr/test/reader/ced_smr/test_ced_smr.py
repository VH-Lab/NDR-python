"""Test reading using NDR reader with CED SMR format.

Ported from +ndr/+test/+reader/+ced_smr/test.m and readertest.m

NOTE: This test requires example data file (example.smr) in the NDR
example_data directory. Download NDR example data to run this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ndr.globals import NDRGlobals


def _get_example_file() -> Path:
    """Return the path to the CED example file."""
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data" / "example.smr"


@pytest.fixture()
def example_smr() -> Path:
    """Provide path to example.smr, skip if not available."""
    f = _get_example_file()
    if not f.exists():
        pytest.skip(
            f"Example data file not found: {f}. "
            "Download NDR example data to run this test."
        )
    return f


def test_reader_creation() -> None:
    """Test that a CED SMR reader can be instantiated."""
    # The MATLAB test calls: r = ndr.reader('smr')
    try:
        from ndr.reader.ced_smr import ndr_reader_ced__smr

        r = ndr_reader_ced__smr()
        assert r is not None
    except ImportError:
        pytest.skip("ndr.reader.ced_smr.ndr_reader_ced__smr not yet available")


def test_getchannelsepoch(example_smr: Path) -> None:
    """Test getting channel information from a CED SMR epoch."""
    # The MATLAB test calls:
    #   r = ndr.reader('smr')
    #   channels = r.getchannelsepoch({filename})
    try:
        from ndr.reader.ced_smr import ndr_reader_ced__smr
    except ImportError:
        pytest.skip("ndr.reader.ced_smr.ndr_reader_ced__smr not yet available")

    r = ndr_reader_ced__smr()
    channels = r.getchannelsepoch([str(example_smr)])

    assert channels is not None
    assert len(channels) > 0

    for ch in channels:
        assert hasattr(ch, "name") or "name" in ch
        assert hasattr(ch, "type") or "type" in ch


def test_readchannels_epochsamples(example_smr: Path) -> None:
    """Test reading epoch samples from a CED SMR file (channel 21)."""
    # The MATLAB test calls:
    #   d = r.readchannels_epochsamples('ai', 21, {filename}, 1, 1, 10000)
    #   t = r.readchannels_epochsamples('time', 21, {filename}, 1, 1, 10000)
    try:
        from ndr.reader.ced_smr import ndr_reader_ced__smr
    except ImportError:
        pytest.skip("ndr.reader.ced_smr.ndr_reader_ced__smr not yet available")

    r = ndr_reader_ced__smr()
    epoch_select = 1
    channel = 21

    d = r.readchannels_epochsamples("ai", channel, [str(example_smr)], epoch_select, 1, 10000)
    t = r.readchannels_epochsamples("time", channel, [str(example_smr)], epoch_select, 1, 10000)

    assert d is not None
    assert t is not None
    assert len(d) == len(t)


def test_epochclock(example_smr: Path) -> None:
    """Test getting epoch clock and t0/t1 from a CED SMR file."""
    try:
        from ndr.reader.ced_smr import ndr_reader_ced__smr
    except ImportError:
        pytest.skip("ndr.reader.ced_smr.ndr_reader_ced__smr not yet available")

    r = ndr_reader_ced__smr()
    epoch_select = 1

    ec = r.epochclock([str(example_smr)], epoch_select)
    t0t1 = r.t0_t1([str(example_smr)], epoch_select)

    assert ec is not None
    assert t0t1 is not None


def test_readertest(example_smr: Path) -> None:
    """Test the reader.read convenience function (from readertest.m).

    Reads analog input channel 21 and event channel 22.
    """
    try:
        from ndr.reader.ced_smr import ndr_reader_ced__smr
    except ImportError:
        pytest.skip("ndr.reader.ced_smr.ndr_reader_ced__smr not yet available")

    r = ndr_reader_ced__smr()

    # Read analog input 21
    d, t = r.read([str(example_smr)], "ai21")
    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)

    # Read event channel 22
    d_e, t_e = r.read([str(example_smr)], "e22")
    assert d_e is not None
    assert t_e is not None
