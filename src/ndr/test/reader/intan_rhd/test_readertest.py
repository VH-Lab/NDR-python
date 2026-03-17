# Port of +ndr/+test/+reader/+intan_rhd/readertest.m
"""Test the functionality of the Intan ndr.reader.read function.

The original MATLAB test creates an Intan reader, then calls r.read()
on the example.rhd file for channel 'A021'.
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
        from ndr.reader.intan_rhd import IntanRHD
    except ImportError:
        pytest.skip("ndr.reader.intan_rhd.IntanRHD not yet available")
    return IntanRHD()


def test_readertest(intan_reader, example_rhd: Path) -> None:
    """Test the Intan reader.read convenience function.

    Reads channel A021 from the example RHD file and verifies data is returned.
    """
    d, t = intan_reader.read([str(example_rhd)], "A021")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
