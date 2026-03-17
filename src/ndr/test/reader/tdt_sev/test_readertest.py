# Port of +ndr/+test/+reader/+tdt_sev/readertest.m
"""Test the functionality of the tdt_sev ndr.reader.read function.

The original MATLAB test creates a TDT SEV reader, then calls r.read()
on the provided filename for channel 'ai1'.
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
        from ndr.reader.tdt_sev import TdtSev
    except ImportError:
        pytest.skip("ndr.reader.tdt_sev.TdtSev not yet available")
    return TdtSev()


@pytest.mark.skip(reason="requires test data download")
def test_readertest(sev_reader, example_sev: Path) -> None:
    """Test the TDT SEV reader.read convenience function.

    Reads channel 'ai1' from the example SEV file and verifies data is returned.
    """
    d, t = sev_reader.read([str(example_sev)], "ai1")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
