# Port of +ndr/+test/+reader/+bjg/readertest.m
"""Test the functionality of the BJG ndr.reader.read function.

The original MATLAB test creates a BJG reader, then calls r.read()
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
def example_bjg() -> Path:
    """Provide path to an example BJG file, skip if not available."""
    example_dir = _get_example_dir()
    for ext in ("*.bjg", "*.bin"):
        files = list(example_dir.glob(ext))
        if files:
            return files[0]
    pytest.skip(
        f"No example BJG files found in {example_dir}. "
        "Download NDR example data to run this test."
    )


@pytest.fixture()
def bjg_reader():
    """Create a BJG reader instance."""
    try:
        from ndr.reader.bjg import BJG
    except ImportError:
        pytest.skip("ndr.reader.bjg.BJG not yet available")
    return BJG()


@pytest.mark.skip(reason="requires test data download")
def test_readertest(bjg_reader, example_bjg: Path) -> None:
    """Test the BJG reader.read convenience function.

    Reads channel 'ai1' from the example BJG file and verifies data is returned.
    """
    d, t = bjg_reader.read([str(example_bjg)], "ai1")

    assert d is not None
    assert t is not None
    assert len(d) > 0
    assert len(d) == len(t)
