"""Test reading using NDR reader with Axon ABF format.

Ported from +ndr/+test/+reader/+axon_abf/test.m and readertest.m

NOTE: This test requires an example ABF data file. The original MATLAB test
expects a filename argument. Download NDR example data to run this test.
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
def example_abf() -> Path:
    """Provide path to an example ABF file, skip if not available."""
    # The MATLAB test takes filename as an argument; look for any .abf in example_data
    example_dir = _get_example_dir()
    abf_files = list(example_dir.glob("*.abf"))
    if not abf_files:
        pytest.skip(
            f"No example ABF files found in {example_dir}. "
            "Download NDR example data to run this test."
        )
    return abf_files[0]


def test_reader_creation() -> None:
    """Test that an ABF reader can be instantiated."""
    # The MATLAB test calls: r = ndr.reader('abf')
    try:
        from ndr.reader.axon_abf import ndr_reader_axon__abf

        r = ndr_reader_axon__abf()
        assert r is not None
    except ImportError:
        pytest.skip("ndr.reader.axon_abf.ndr_reader_axon__abf not yet available")


def test_getchannelsepoch(example_abf: Path) -> None:
    """Test getting channel information from an ABF epoch."""
    # The MATLAB test calls:
    #   r = ndr.reader('abf')
    #   channels = r.getchannelsepoch({filename})
    try:
        from ndr.reader.axon_abf import ndr_reader_axon__abf
    except ImportError:
        pytest.skip("ndr.reader.axon_abf.ndr_reader_axon__abf not yet available")

    r = ndr_reader_axon__abf()
    channels = r.getchannelsepoch([str(example_abf)])

    assert channels is not None
    assert len(channels) > 0

    # Each channel should have a name and type
    for ch in channels:
        assert hasattr(ch, "name") or "name" in ch
        assert hasattr(ch, "type") or "type" in ch


def test_readchannels_epochsamples(example_abf: Path) -> None:
    """Test reading epoch samples from an ABF file."""
    # The MATLAB test calls:
    #   d = r.readchannels_epochsamples('ai', 1, {filename}, 1, 1, 10000)
    #   t = r.readchannels_epochsamples('time', 1, {filename}, 1, 1, 10000)
    try:
        from ndr.reader.axon_abf import ndr_reader_axon__abf
    except ImportError:
        pytest.skip("ndr.reader.axon_abf.ndr_reader_axon__abf not yet available")

    r = ndr_reader_axon__abf()
    epoch_select = 1
    channel = 1

    d = r.readchannels_epochsamples("ai", channel, [str(example_abf)], epoch_select, 1, 10000)
    t = r.readchannels_epochsamples("time", channel, [str(example_abf)], epoch_select, 1, 10000)

    assert d is not None
    assert t is not None
    assert len(d) == len(t)


def test_epochclock(example_abf: Path) -> None:
    """Test getting epoch clock and t0/t1 from an ABF file."""
    # The MATLAB test calls:
    #   ec = r.epochclock({filename}, epoch_select)
    #   t0t1 = r.t0_t1({filename}, epoch_select)
    try:
        from ndr.reader.axon_abf import ndr_reader_axon__abf
    except ImportError:
        pytest.skip("ndr.reader.axon_abf.ndr_reader_axon__abf not yet available")

    r = ndr_reader_axon__abf()
    epoch_select = 1

    ec = r.epochclock([str(example_abf)], epoch_select)
    t0t1 = r.t0_t1([str(example_abf)], epoch_select)

    assert ec is not None
    assert t0t1 is not None


def test_readertest(example_abf: Path) -> None:
    """Test the reader.read convenience function (from readertest.m)."""
    # The MATLAB readertest calls:
    #   r = ndr.reader('axon_abf')
    #   [d, t] = r.read({filename}, 'ai1')
    try:
        from ndr.reader.axon_abf import ndr_reader_axon__abf
    except ImportError:
        pytest.skip("ndr.reader.axon_abf.ndr_reader_axon__abf not yet available")

    r = ndr_reader_axon__abf()
    d, t = r.read([str(example_abf)], "ai1")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
