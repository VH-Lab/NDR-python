"""Automated tests for the Neo reader.

Ported from +ndr/+test/+reader/+neo/automatedTest.m

This test covers Neo reader functionality with Blackrock (.ns2) and Intan
(.rhd) file formats, including reading markers, events, analog data, and
channel epoch information.

NOTE: This test requires example data files (example_1.ns2, example_2.ns2,
example.rhd, example.smr) in the NDR example_data directory. Download NDR
example data to run this test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ndr.globals import NDRGlobals


def _get_example(filename: str) -> Path:
    """Return the path to an NDR example file."""
    g = NDRGlobals()
    return Path(g.path["path"]) / "example_data" / filename


def _skip_if_missing(filepath: Path) -> Path:
    """Skip the test if the example file is not present."""
    if not filepath.exists():
        pytest.skip(
            f"Example data file not found: {filepath}. "
            "Download NDR example data to run this test."
        )
    return filepath


@pytest.fixture()
def neo_reader():
    """Create a Neo reader instance."""
    try:
        from ndr.reader.neo import Neo
    except ImportError:
        pytest.skip("ndr.reader.neo.Neo not yet available")
    return Neo()


@pytest.fixture()
def intan_reader():
    """Create an Intan RHD reader instance."""
    try:
        from ndr.reader.intan_rhd import IntanRHD
    except ImportError:
        pytest.skip("ndr.reader.intan_rhd.IntanRHD not yet available")
    return IntanRHD()


@pytest.fixture()
def ced_reader():
    """Create a CED SMR reader instance."""
    try:
        from ndr.reader.ced_smr import CedSMR
    except ImportError:
        pytest.skip("ndr.reader.ced_smr.CedSMR not yet available")
    return CedSMR()


# ---------------------------------------------------------------------------
# Blackrock event/marker tests
# ---------------------------------------------------------------------------


class TestReadeventsBlackrock:
    """Tests for readevents_epochsamples_native with Blackrock files."""

    @pytest.fixture()
    def example_ns2(self) -> Path:
        return _skip_if_missing(_get_example("example_1.ns2"))

    def test_marker_many_channels(self, neo_reader, example_ns2: Path) -> None:
        """Read 'marker' from multiple channels in a Blackrock file."""
        timestamps, data = neo_reader.readevents_epochsamples_native(
            "marker",
            ["digital_input_port", "serial_input_port", "analog_input_channel_1"],
            [str(example_ns2)],
            1,
            0,
            100,
        )

        # Structure checks
        assert len(timestamps) == 3
        assert isinstance(timestamps, list)
        assert len(data) == 3
        assert isinstance(data, list)

        # Value checks for first channel (digital_input_port)
        expected_t = np.array([0.1349, 0.1385, 0.1604, 0.5421, 0.9435, 1.2480, 2.2508, 2.4426])
        np.testing.assert_allclose(timestamps[0], expected_t, atol=0.001)

        expected_d = ["65280", "65296", "65280", "65344", "65349", "65344", "65350", "65382"]
        assert list(data[0]) == expected_d

    def test_marker_single_channel(self, neo_reader, example_ns2: Path) -> None:
        """Read 'marker' from a single channel in a Blackrock file."""
        timestamps, data = neo_reader.readevents_epochsamples_native(
            "marker",
            ["digital_input_port"],
            [str(example_ns2)],
            1,
            0,
            100,
        )

        # Single channel: should return arrays directly
        assert isinstance(timestamps, np.ndarray)
        assert timestamps.shape == (8,)
        assert len(data) == 8

        expected_t = np.array([0.1349, 0.1385, 0.1604, 0.5421, 0.9435, 1.2480, 2.2508, 2.4426])
        np.testing.assert_allclose(timestamps, expected_t, atol=0.001)

        expected_d = ["65280", "65296", "65280", "65344", "65349", "65344", "65350", "65382"]
        assert list(data) == expected_d

    def test_event_many_channels(self, neo_reader, example_ns2: Path) -> None:
        """Read 'event' from multiple channels in a Blackrock file."""
        timestamps, data = neo_reader.readevents_epochsamples_native(
            "event",
            ["ch1#0", "ch1#255"],
            [str(example_ns2)],
            1,
            0,
            0.4,
        )

        # Structure checks
        assert len(timestamps) == 2
        assert isinstance(timestamps, list)
        assert len(data) == 2
        assert isinstance(data, list)

        # Value checks for second channel (ch1#255)
        np.testing.assert_allclose(timestamps[1], np.array([0.2761, 0.3508]), atol=0.001)
        assert list(data[1]) == ["1", "1"]

    def test_event_single_channel(self, neo_reader, example_ns2: Path) -> None:
        """Read 'event' from a single channel in a Blackrock file."""
        timestamps, data = neo_reader.readevents_epochsamples_native(
            "event",
            ["ch1#255"],
            [str(example_ns2)],
            1,
            0,
            0.4,
        )

        np.testing.assert_allclose(timestamps, np.array([0.2761, 0.3508]), atol=0.001)
        assert list(data) == ["1", "1"]


# ---------------------------------------------------------------------------
# Cross-reader comparison: Intan vs Neo
# ---------------------------------------------------------------------------


class TestCrossReaderIntan:
    """Compare Intan and Neo readers on the same Intan data."""

    @pytest.fixture()
    def example_rhd(self) -> Path:
        return _skip_if_missing(_get_example("example.rhd"))

    def test_read_intan(self, intan_reader, neo_reader, example_rhd: Path) -> None:
        """Compare read() output between Intan and Neo readers."""
        intan_data, intan_time = intan_reader.read(
            [str(example_rhd)],
            "A000+A001",
            {"useSamples": 1, "s0": 5, "s1": 8},
        )

        neo_data, neo_time = neo_reader.read(
            [str(example_rhd)],
            ["A-000", "A-001"],
            {"useSamples": 1, "s0": 5, "s1": 8},
        )

        np.testing.assert_allclose(intan_data, neo_data, atol=0.001)
        np.testing.assert_allclose(intan_time, neo_time, atol=0.001)

    def test_readchannels_epochsamples_intan(
        self, intan_reader, neo_reader, example_rhd: Path
    ) -> None:
        """Compare readchannels_epochsamples for analog and time channels."""
        # Analog input
        intan_data = intan_reader.readchannels_epochsamples(
            "ai", [1, 2], [str(example_rhd)], 1, 1, 10
        )
        neo_data = neo_reader.readchannels_epochsamples(
            "smth", ["A-000", "A-001"], [str(example_rhd)], 1, 1, 10
        )
        np.testing.assert_allclose(intan_data, neo_data, atol=0.001)

        # Time
        intan_time = intan_reader.readchannels_epochsamples(
            "time", [1, 2], [str(example_rhd)], 1, 1, 10
        )
        neo_time = neo_reader.readchannels_epochsamples(
            "time", ["A-000", "A-001"], [str(example_rhd)], 1, 1, 10
        )
        np.testing.assert_allclose(intan_time, neo_time, atol=1e-7)


# ---------------------------------------------------------------------------
# Channel epoch tests
# ---------------------------------------------------------------------------


class TestGetchannelsepoch:
    """Tests for getchannelsepoch across different file formats."""

    def test_blackrock_example2(self, neo_reader) -> None:
        """Test channel listing for example_2.ns2."""
        filepath = _skip_if_missing(_get_example("example_2.ns2"))

        channels = neo_reader.getchannelsepoch([str(filepath)], "all")

        assert len(channels) == 6
        # First two channels
        assert channels[0] == {"name": "ainp9", "type": "analog_input"}
        assert channels[1] == {"name": "ainp10", "type": "analog_input"}

    def test_blackrock_example1(self, neo_reader) -> None:
        """Test channel listing for example_1.ns2."""
        filepath = _skip_if_missing(_get_example("example_1.ns2"))

        channels = neo_reader.getchannelsepoch([str(filepath)], "all")

        assert len(channels) == 332
        assert channels[0] == {"name": "ch1#0", "type": "event"}
        assert channels[1] == {"name": "ch1#255", "type": "event"}
        assert channels[2] == {"name": "ch2#0", "type": "event"}
        assert channels[324] == {"name": "digital_input_port", "type": "marker"}
        assert channels[325] == {"name": "serial_input_port", "type": "marker"}
        assert channels[326] == {"name": "analog_input_channel_1", "type": "marker"}

    def test_intan_channels(self, intan_reader, neo_reader) -> None:
        """Compare channel counts between Intan and Neo readers for .rhd."""
        filepath = _skip_if_missing(_get_example("example.rhd"))

        intan_channels = intan_reader.getchannelsepoch([str(filepath)], 1)
        neo_channels = neo_reader.getchannelsepoch([str(filepath)], "all")

        # Different channel naming conventions
        assert intan_channels[0]["name"] == "ai1" or getattr(intan_channels[0], "name", None) == "ai1"
        assert neo_channels[0]["name"] == "A-000" or getattr(neo_channels[0], "name", None) == "A-000"

        # Same number of channels
        assert len(intan_channels) == 36
        assert len(neo_channels) == 36

    def test_ced_channels(self, ced_reader, neo_reader) -> None:
        """Compare channel counts between CED and Neo readers for .smr."""
        filepath = _skip_if_missing(_get_example("example.smr"))

        ced_channels = ced_reader.getchannelsepoch([str(filepath)], 1)
        neo_channels = neo_reader.getchannelsepoch([str(filepath)], "all")

        # Neo returns fewer channels than CED
        assert len(ced_channels) == 14
        assert len(neo_channels) == 3


# ---------------------------------------------------------------------------
# Blackrock analog read tests
# ---------------------------------------------------------------------------


class TestReadchannelsBlackrock:
    """Tests for readchannels_epochsamples with Blackrock files."""

    def test_analog_input(self, neo_reader) -> None:
        """Test reading analog input channels from example_2.ns2."""
        filepath = _skip_if_missing(_get_example("example_2.ns2"))

        data = neo_reader.readchannels_epochsamples(
            "smth", ["ainp9", "ainp10"], [str(filepath)], 1, 1, 3
        )

        expected = np.array([[137, 761], [125, 747], [110, 733]])
        np.testing.assert_array_equal(data, expected)

    def test_time_channel(self, neo_reader) -> None:
        """Test reading time channel from example_2.ns2."""
        filepath = _skip_if_missing(_get_example("example_2.ns2"))

        time = neo_reader.readchannels_epochsamples(
            "time", ["ainp9", "ainp10"], [str(filepath)], 1, 1, 3
        )

        expected = np.array([0.0, 0.001, 0.002])
        np.testing.assert_allclose(time, expected, atol=1e-7)
