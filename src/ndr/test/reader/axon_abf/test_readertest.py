# Port of +ndr/+test/+reader/+axon_abf/readertest.m
"""Test the functionality of the axon_abf ndr.reader.read function.

The original MATLAB test creates an ABF reader, then calls r.read()
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
def example_abf() -> Path:
    """Provide path to an example ABF file, skip if not available."""
    example_dir = _get_example_dir()
    abf_files = list(example_dir.glob("*.abf"))
    if not abf_files:
        pytest.skip(
            f"No example ABF files found in {example_dir}. "
            "Download NDR example data to run this test."
        )
    return abf_files[0]


@pytest.fixture()
def abf_reader():
    """Create an Axon ABF reader instance."""
    try:
        from ndr.reader.axon_abf import ndr_reader_axon__abf
    except ImportError:
        pytest.skip("ndr.reader.axon_abf.ndr_reader_axon__abf not yet available")
    return ndr_reader_axon__abf()


def test_readertest(abf_reader, example_abf: Path) -> None:
    """Test the ABF reader.read convenience function.

    Reads channel 'ai1' from the example ABF file and verifies data is returned.
    """
    d, t = abf_reader.read([str(example_abf)], "ai1")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
