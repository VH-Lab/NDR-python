"""Test reading using NDR reader with BJG bin format.

Ported from +ndr/+test/+reader/+bjg/test.m and readertest.m

NOTE: This test requires an example BJG data file. The original MATLAB test
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
def example_bjg() -> Path:
    """Provide path to an example BJG file, skip if not available."""
    example_dir = _get_example_dir()
    # BJG files may have various extensions; look for common ones
    for ext in ("*.bjg", "*.bin"):
        files = list(example_dir.glob(ext))
        if files:
            return files[0]
    pytest.skip(
        f"No example BJG files found in {example_dir}. "
        "Download NDR example data to run this test."
    )


def test_reader_creation() -> None:
    """Test that a BJG reader can be instantiated."""
    # The MATLAB test calls: r = ndr.reader('bjg')
    try:
        from ndr.reader.bjg import BJG

        r = BJG()
        assert r is not None
    except ImportError:
        pytest.skip("ndr.reader.bjg.BJG not yet available")


def test_getchannelsepoch(example_bjg: Path) -> None:
    """Test getting channel information from a BJG epoch."""
    # The MATLAB test calls:
    #   r = ndr.reader('bjg')
    #   channels = r.getchannelsepoch({filename})
    try:
        from ndr.reader.bjg import BJG
    except ImportError:
        pytest.skip("ndr.reader.bjg.BJG not yet available")

    r = BJG()
    channels = r.getchannelsepoch([str(example_bjg)])

    assert channels is not None
    assert len(channels) > 0

    for ch in channels:
        assert hasattr(ch, "name") or "name" in ch
        assert hasattr(ch, "type") or "type" in ch


def test_readchannels_epochsamples(example_bjg: Path) -> None:
    """Test reading epoch samples from a BJG file."""
    # The MATLAB test calls:
    #   d = r.readchannels_epochsamples('ai', 1, {filename}, 1, 1, 10000)
    #   t = r.readchannels_epochsamples('time', 1, {filename}, 1, 1, 10000)
    try:
        from ndr.reader.bjg import BJG
    except ImportError:
        pytest.skip("ndr.reader.bjg.BJG not yet available")

    r = BJG()
    epoch_select = 1
    channel = 1

    d = r.readchannels_epochsamples("ai", channel, [str(example_bjg)], epoch_select, 1, 10000)
    t = r.readchannels_epochsamples("time", channel, [str(example_bjg)], epoch_select, 1, 10000)

    assert d is not None
    assert t is not None
    assert len(d) == len(t)


def test_epochclock(example_bjg: Path) -> None:
    """Test getting epoch clock and t0/t1 from a BJG file."""
    try:
        from ndr.reader.bjg import BJG
    except ImportError:
        pytest.skip("ndr.reader.bjg.BJG not yet available")

    r = BJG()
    epoch_select = 1

    ec = r.epochclock([str(example_bjg)], epoch_select)
    t0t1 = r.t0_t1([str(example_bjg)], epoch_select)

    assert ec is not None
    assert t0t1 is not None


def test_readertest(example_bjg: Path) -> None:
    """Test the reader.read convenience function (from readertest.m)."""
    # The MATLAB readertest calls:
    #   r = ndr.reader('bjg')
    #   [d, t] = r.read({filename}, 'ai1')
    try:
        from ndr.reader.bjg import BJG
    except ImportError:
        pytest.skip("ndr.reader.bjg.BJG not yet available")

    r = BJG()
    d, t = r.read([str(example_bjg)], "ai1")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
