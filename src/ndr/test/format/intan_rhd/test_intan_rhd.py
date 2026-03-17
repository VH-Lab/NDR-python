"""Test reading from Intan RHD format.

Ported from +ndr/+test/+format/+intan_rhd/test.m

NOTE: This test requires example data files (example.rhd) that must be
downloaded or placed in the NDR example_data directory before running.
The original MATLAB test uses ndr_globals.path.path/example_data/example.rhd.
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


def test_read_header(example_rhd: Path) -> None:
    """Test reading the Intan RHD2000 header."""
    # The MATLAB test calls:
    #   h = ndr.format.intan.read_Intan_RHD2000_header(filename)
    try:
        from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header
    except ImportError:
        pytest.skip("ndr.format.intan.read_Intan_RHD2000_header not yet available")

    h = read_Intan_RHD2000_header(str(example_rhd))
    assert h is not None


def test_read_datafile(example_rhd: Path) -> None:
    """Test reading time and amplitude data from Intan RHD file."""
    # The MATLAB test calls:
    #   t = ndr.format.intan.read_Intan_RHD2000_datafile(filename, h, 'time', 1, 0, 100)
    #   d = ndr.format.intan.read_Intan_RHD2000_datafile(filename, h, 'amp', 1, 0, 100)
    try:
        from ndr.format.intan.read_Intan_RHD2000_datafile import read_Intan_RHD2000_datafile
        from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header
    except ImportError:
        pytest.skip("ndr.format.intan read functions not yet available")

    h = read_Intan_RHD2000_header(str(example_rhd))
    t = read_Intan_RHD2000_datafile(str(example_rhd), h, "time", 1, 0, 100)
    d = read_Intan_RHD2000_datafile(str(example_rhd), h, "amp", 1, 0, 100)

    assert t is not None
    assert d is not None
    assert len(t) > 0
    assert len(d) > 0
