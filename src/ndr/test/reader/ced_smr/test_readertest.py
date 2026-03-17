# Port of +ndr/+test/+reader/+ced_smr/readertest.m
"""Test the functionality of the CED ndr.reader.read function.

The original MATLAB test creates a CED SMR reader, reads analog input
channel 21 and event channel 22 from example.smr.
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


@pytest.fixture()
def smr_reader():
    """Create a CED SMR reader instance."""
    try:
        from ndr.reader.ced_smr import ndr_reader_ced__smr
    except ImportError:
        pytest.skip("ndr.reader.ced_smr.ndr_reader_ced__smr not yet available")
    return ndr_reader_ced__smr()


def test_readertest(smr_reader, example_smr: Path) -> None:
    """Test the CED SMR reader.read convenience function.

    Reads analog-input channel 21 and event channel 22 from the example
    SMR file.
    """
    # Read analog input channel 21
    d, t = smr_reader.read([str(example_smr)], "ai21")
    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)

    # Read event channel 22
    d_e, t_e = smr_reader.read([str(example_smr)], "e22")
    assert d_e is not None
    assert t_e is not None
