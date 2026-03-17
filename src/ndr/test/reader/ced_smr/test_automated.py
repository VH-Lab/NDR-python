"""Automated tests for CED SMR reader.

Ported from +ndr/+test/+reader/+ced_smr/automatedTest.m

This is a detailed automated test with specific expected values for the
CED SMR reader's readevents_epochsamples_native method.

NOTE: This test requires example data file (example.smr) in the NDR
example_data directory. Download NDR example data to run this test.

Channel layout in example.smr:
  Channel  1/14: ai1    - analog_in
  Channel  2/14: mk20   - mark
  Channel  3/14: ai21   - analog_in
  Channel  4/14: e22    - event
  Channel  5/14: e23    - event
  Channel  6/14: e24    - event
  Channel  7/14: e25    - event
  Channel  8/14: e26    - event
  Channel  9/14: e27    - event
  Channel 10/14: e28    - event
  Channel 11/14: e29    - event
  Channel 12/14: text30 - text
  Channel 13/14: mk31   - mark
  Channel 14/14: mk32   - mark
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
            f"Example data file not found: {f}. " "Download NDR example data to run this test."
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


class TestReadeventsMarker:
    """Tests for readevents_epochsamples_native with marker channels."""

    def test_marker_many_channels(self, smr_reader, example_smr: Path) -> None:
        """Read 'marker' type from multiple channels (30, 31, 32)."""
        timestamps, data = smr_reader.readevents_epochsamples_native(
            "marker", [30, 31, 32], [str(example_smr)], 1, 0, 100
        )

        # Data structure checks
        assert len(timestamps) == 3
        assert isinstance(timestamps, list)
        assert len(data) == 3
        assert isinstance(data, list)

        # Expected timestamps
        t_expected = [
            np.array(
                [
                    9.46626,
                    20.25515,
                    31.02761,
                    41.80011,
                    52.57258,
                    63.35325,
                    74.13393,
                    84.91459,
                    95.67888,
                ]
            ),
            np.array(
                [
                    12.36624,
                    23.14691,
                    33.91940,
                    44.70006,
                    55.47256,
                    66.25321,
                    77.02569,
                    87.80637,
                    98.57885,
                ]
            ),
            np.array(
                [
                    9.46533,
                    20.23972,
                    31.01611,
                    41.79250,
                    52.56889,
                    63.34528,
                    74.12267,
                    84.89906,
                    95.67545,
                ]
            ),
        ]

        for i in range(3):
            np.testing.assert_allclose(
                timestamps[i],
                t_expected[i],
                atol=1e-4,
                err_msg=f"Timestamps mismatch for channel index {i}",
            )

    def test_marker_single_channel(self, smr_reader, example_smr: Path) -> None:
        """Read 'marker' type from a single channel (30)."""
        timestamps, data = smr_reader.readevents_epochsamples_native(
            "marker", [30], [str(example_smr)], 1, 0, 100
        )

        # When single channel, should return arrays directly (not nested in list)
        assert isinstance(timestamps, np.ndarray)
        assert timestamps.shape == (9,)

        t_expected = np.array(
            [
                9.46626,
                20.25515,
                31.02761,
                41.80011,
                52.57258,
                63.35325,
                74.13393,
                84.91459,
                95.67888,
            ]
        )
        np.testing.assert_allclose(timestamps, t_expected, atol=1e-4)


class TestReadeventsEvent:
    """Tests for readevents_epochsamples_native with event channels."""

    def test_event_many_channels(self, smr_reader, example_smr: Path) -> None:
        """Read 'event' type from multiple channels (22, 23, 28, 29)."""
        timestamps, data = smr_reader.readevents_epochsamples_native(
            "event", [22, 23, 28, 29], [str(example_smr)], 1, 0, 10
        )

        # Data structure checks
        assert len(timestamps) == 4
        assert isinstance(timestamps, list)
        assert len(data) == 4
        assert isinstance(data, list)

        # Channel 22 (index 0): single event at ~9.46533
        t0_expected = np.array([9.46533])
        np.testing.assert_allclose(timestamps[0], t0_expected, atol=1e-4)

        # Channel 23 (index 1): many events
        t1_expected = np.array(
            [
                9.47933,
                9.49633,
                9.51233,
                9.52933,
                9.54533,
                9.56233,
                9.57933,
                9.59533,
                9.61233,
                9.62833,
                9.64533,
                9.66133,
                9.67833,
                9.69533,
                9.71133,
                9.72834,
                9.74434,
                9.76134,
                9.77734,
                9.79434,
                9.81134,
                9.82734,
                9.84434,
                9.86034,
                9.87734,
                9.89434,
                9.91034,
                9.92734,
                9.94334,
                9.96034,
                9.97634,
                9.99334,
            ]
        )
        np.testing.assert_allclose(timestamps[1], t1_expected, atol=1e-4)

        # Channels 28, 29 (indices 2, 3): empty
        assert len(timestamps[2]) == 0
        assert len(timestamps[3]) == 0

        # For event channels, data should equal timestamps
        np.testing.assert_allclose(timestamps[0], data[0], atol=1e-10)
        np.testing.assert_allclose(timestamps[1], data[1], atol=1e-10)

    def test_event_single_channel(self, smr_reader, example_smr: Path) -> None:
        """Read 'event' type from a single channel (23)."""
        timestamps, data = smr_reader.readevents_epochsamples_native(
            "event", [23], [str(example_smr)], 1, 0, 10
        )

        t_expected = np.array(
            [
                9.47933,
                9.49633,
                9.51233,
                9.52933,
                9.54533,
                9.56233,
                9.57933,
                9.59533,
                9.61233,
                9.62833,
                9.64533,
                9.66133,
                9.67833,
                9.69533,
                9.71133,
                9.72834,
                9.74434,
                9.76134,
                9.77734,
                9.79434,
                9.81134,
                9.82734,
                9.84434,
                9.86034,
                9.87734,
                9.89434,
                9.91034,
                9.92734,
                9.94334,
                9.96034,
                9.97634,
                9.99334,
            ]
        )
        np.testing.assert_allclose(timestamps, t_expected, atol=1e-4)
        np.testing.assert_allclose(data, t_expected, atol=1e-4)
