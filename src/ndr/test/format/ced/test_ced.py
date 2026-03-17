"""Test reading from CED format.

Ported from +ndr/+test/+format/+ced/test.m

NOTE: This test requires example data files (example.smr) that must be
downloaded or placed in the NDR example_data directory before running.
The original MATLAB test uses ndr_globals.path.path/example_data/example.smr.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ndr.globals import NDRGlobals


def _get_example_dir() -> Path:
    """Return the path to the NDR example_data directory."""
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data"


def _get_example_file() -> Path:
    """Return the path to the CED example file."""
    return _get_example_dir() / "example.smr"


@pytest.fixture()
def example_smr() -> Path:
    """Provide path to example.smr, skip if not available."""
    f = _get_example_file()
    if not f.exists():
        pytest.skip(
            f"Example data file not found: {f}. " "Download NDR example data to run this test."
        )
    return f


def test_read_header(example_smr: Path) -> None:
    """Test reading the CED SMR header."""
    # TODO: Port ndr.format.ced.read_SOMSMR_header to Python
    # h = ndr.format.ced.read_SOMSMR_header(example_smr)
    # assert h is not None
    # assert hasattr(h, 'fileinfo')
    # assert hasattr(h, 'channelinfo')
    pytest.skip("ndr.format.ced.read_SOMSMR_header not yet ported to Python")


def test_read_datafile(example_smr: Path) -> None:
    """Test reading data from CED SMR file (channel 21, 0-100s)."""
    # TODO: Port ndr.format.ced.read_SOMSMR_datafile to Python
    # h = ndr.format.ced.read_SOMSMR_header(example_smr)
    # data, total_samples, total_time, blockinfo, time = (
    #     ndr.format.ced.read_SOMSMR_datafile(example_smr, h, 21, 0, 100)
    # )
    # assert data is not None
    # assert len(data) > 0
    pytest.skip("ndr.format.ced.read_SOMSMR_datafile not yet ported to Python")


def test_read_sampleinterval(example_smr: Path) -> None:
    """Test reading sample interval from CED SMR file."""
    # TODO: Port ndr.format.ced.read_SOMSMR_sampleinterval to Python
    # h = ndr.format.ced.read_SOMSMR_header(example_smr)
    # interval = ndr.format.ced.read_SOMSMR_sampleinterval(example_smr, h, 21)
    # sample_rate = 1.0 / interval
    # assert sample_rate > 0
    pytest.skip("ndr.format.ced.read_SOMSMR_sampleinterval not yet ported to Python")
