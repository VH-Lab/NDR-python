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


def _require_sonpipe():
    """Skip unless the real sonpipe CLI is installed.

    CED's sonpy has no wheel for CPython 3.10-3.13 on Linux or macOS, so on the
    main test matrix the CLI cannot be present. The ced-integration CI job runs
    this file on 3.14 with sonpipe installed; tests/test_ced_sonpipe.py covers
    the bridge itself on every version via a stand-in CLI.
    """
    import os

    from ndr.format.ced import sonpipe

    sonpipe.reset_cache()
    try:
        sonpipe.executable()
    except sonpipe.SonpipeNotFoundError:
        if os.environ.get("NDR_REQUIRE_SONPIPE"):
            # The ced-integration job sets this. Skipping there would make the
            # job pass without reading a single byte through the real CLI --
            # exactly the vacuous green this suite exists to avoid.
            raise
        pytest.skip("sonpipe CLI not installed; see ndr.format.ced.sonpipe.executable")


def test_read_header(example_smr: Path) -> None:
    """Read the CED SMR header through the real sonpipe CLI."""
    _require_sonpipe()
    from ndr.format.ced.read_SOMSMR_header import read_SOMSMR_header

    h = read_SOMSMR_header(example_smr)
    assert h["fileinfo"]["timebase"] > 0
    assert h["channelinfo"], "example.smr should record at least one channel"
    # The classic-SON aliases read_SOMSMR_header.m adds.
    assert h["fileinfo"]["dTimeBase"] == h["fileinfo"]["timebase"]
    assert h["fileinfo"]["usPerTime"] == 1
    for channel in h["channelinfo"]:
        assert channel["ndr_type"] in ("analog_in", "event", "mark", "text", "unknown")


def test_read_datafile(example_smr: Path) -> None:
    """Read a waveform channel and check it lines up with its own time base."""
    _require_sonpipe()
    from ndr.format.ced.read_SOMSMR_datafile import read_SOMSMR_datafile
    from ndr.format.ced.read_SOMSMR_header import read_SOMSMR_header

    h = read_SOMSMR_header(example_smr)
    waveform = next((c for c in h["channelinfo"] if c["kind"] in (1, 9)), None)
    if waveform is None:
        pytest.skip("example.smr records no waveform channel")

    data, total_samples, total_time, blockinfo, time = read_SOMSMR_datafile(
        example_smr, h, waveform["number"], 0, 1
    )
    assert data.ndim == 2 and data.shape[1] == 1
    assert data.shape[0] > 0
    assert data.shape == time.shape
    assert total_samples == waveform["num_samples"]
    assert blockinfo is None
    # One second of data at this channel's rate -- or the whole channel, if it
    # holds less than a second. Sample 0 and sample sr both fall inside [0, 1],
    # hence sr + 1.
    expected = min(waveform["samplerate"] + 1, waveform["num_samples"])
    assert abs(data.shape[0] - expected) <= 1


def test_read_sampleinterval(example_smr: Path) -> None:
    """The sample interval must agree with the header's own sample rate."""
    _require_sonpipe()
    from ndr.format.ced.read_SOMSMR_header import read_SOMSMR_header
    from ndr.format.ced.read_SOMSMR_sampleinterval import read_SOMSMR_sampleinterval

    h = read_SOMSMR_header(example_smr)
    waveform = next((c for c in h["channelinfo"] if c["kind"] in (1, 9)), None)
    if waveform is None:
        pytest.skip("example.smr records no waveform channel")

    interval, total_samples, total_time = read_SOMSMR_sampleinterval(
        example_smr, h, waveform["number"]
    )
    assert interval > 0
    assert 1.0 / interval == pytest.approx(waveform["samplerate"])
    assert total_samples == waveform["num_samples"]
    assert total_time > 0
