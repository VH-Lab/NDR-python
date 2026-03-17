"""Test reading from SpikeGadgets format.

Ported from +ndr/+test/+format/+spikegadgets/test.m

NOTE: This test requires example data files (example.rec) that must be
downloaded or placed in the NDR example_data directory before running.
The original MATLAB test uses ndr_globals.path.path/example_data/example.rec.
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
            f"Example data file not found: {f}. " "Download NDR example data to run this test."
        )
    return f


def test_read_config(example_rec: Path) -> None:
    """Test reading the SpikeGadgets .rec header/config."""
    # The MATLAB test calls:
    #   h = ndr.format.spikegadgets.read_rec_config(filename)
    try:
        from ndr.format.spikegadgets.read_rec_config import read_rec_config
    except ImportError:
        pytest.skip("ndr.format.spikegadgets.read_rec_config not yet available")

    h = read_rec_config(str(example_rec))
    assert h is not None
    assert hasattr(h, "samplingRate") or "samplingRate" in h


def test_read_trode_channels(example_rec: Path) -> None:
    """Test reading trode channel data from SpikeGadgets .rec file."""
    # The MATLAB test calls:
    #   t1 = 1 * eval(h.samplingRate)
    #   [data, time] = ndr.format.spikegadgets.read_rec_trodeChannels(
    #       filename, h.numChannels, 1, eval(h.samplingRate), h.headerSize, 1, t1)
    try:
        from ndr.format.spikegadgets.read_rec_config import read_rec_config
        from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels
    except ImportError:
        pytest.skip("ndr.format.spikegadgets read functions not yet available")

    h = read_rec_config(str(example_rec))
    sampling_rate = int(h["samplingRate"]) if isinstance(h, dict) else int(h.samplingRate)
    num_channels = int(h["numChannels"]) if isinstance(h, dict) else int(h.numChannels)
    header_size = int(h["headerSize"]) if isinstance(h, dict) else int(h.headerSize)

    t1 = 1 * sampling_rate
    data, time = read_rec_trodeChannels(
        str(example_rec), num_channels, 1, sampling_rate, header_size, 1, t1
    )

    assert data is not None
    assert time is not None
    assert len(data) > 0
