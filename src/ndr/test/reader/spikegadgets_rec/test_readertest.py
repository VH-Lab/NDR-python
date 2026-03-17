# Port of +ndr/+test/+reader/+spikegadgets_rec/readertest.m
"""Test the functionality of the SpikeGadgets rec ndr.reader.read function.

The original MATLAB test creates a SpikeGadgets rec reader, then calls
r.read() on the example.rec file for channel 'ai120'.
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


@pytest.mark.xfail(reason="SpikeGadgets reader not yet fully implemented")
def test_readertest(rec_reader, example_rec: Path) -> None:
    """Test the SpikeGadgets rec reader.read convenience function.

    Reads channel 'ai120' from the example.rec file and verifies data
    is returned.
    """
    d, t = rec_reader.read([str(example_rec)], "ai120")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
