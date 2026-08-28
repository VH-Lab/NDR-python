"""Verify SpikeGadgets format-layer symmetry artifacts.

Parameterized over both artifact sources, so one test checks parity in both
directions. Skips when the artifact directory for a source is absent.

Tolerances are tight on purpose: these are int16 codes scaled by a constant,
not measurements, so the two ports should agree to floating-point noise. A
loose tolerance here would hide exactly the class of bug this exists to catch
-- an offset or stride error yields plausible in-range values, not obviously
broken ones.

Mirrors tools/tests/+ndr/+symmetry/+readArtifacts/+format/readSpikeGadgets.m
"""

import json

import numpy as np
import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, SOURCE_TYPES, SYMMETRY_BASE


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadSpikeGadgets:
    def _artifact_dir(self, source_type):
        return (
            SYMMETRY_BASE
            / source_type
            / "format"
            / "readSpikeGadgets"
            / "testReadSpikeGadgetsArtifacts"
        )

    def _config(self):
        rec = EXAMPLE_DATA / "example.rec"
        if not rec.exists():
            pytest.skip("Example .rec file not available")

        from ndr.format.spikegadgets.read_rec_config import read_rec_config

        config, _channels = read_rec_config(str(rec))
        return (
            rec,
            int(config["numChannels"]),
            int(config["headerSize"]),
            float(config["samplingRate"]),
        )

    def test_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "metadata.json").read_text())
        _rec, num_channels, header_size, sampling_rate = self._config()

        assert num_channels == expected["numChannels"]
        assert header_size == expected["headerSize"]
        assert sampling_rate == pytest.approx(expected["samplingRate"], abs=1e-9)

    def test_samples(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "readData.json").read_text())
        rec, num_channels, header_size, sampling_rate = self._config()

        from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels

        data1, ts1 = read_rec_trodeChannels(
            rec, num_channels, [1], sampling_rate, header_size, 1, 100
        )
        np.testing.assert_allclose(
            np.asarray(data1).ravel(),
            expected["trode_channel_1_samples_1_100"],
            atol=1e-9,
            err_msg=(
                "Trode channel 1 samples 1-100 differ: the ports disagree on where "
                "packet data begins or how wide a packet is."
            ),
        )
        np.testing.assert_allclose(
            np.asarray(ts1).ravel(), expected["trode_timestamps_samples_1_100"], atol=1e-9
        )

        data2, _ = read_rec_trodeChannels(
            rec, num_channels, [1], sampling_rate, header_size, 1001, 1100
        )
        np.testing.assert_allclose(
            np.asarray(data2).ravel(),
            expected["trode_channel_1_samples_1001_1100"],
            atol=1e-9,
            err_msg=(
                "Samples 1001-1100 differ: an error in the seek to s0 scales with "
                "s0, so this can fail while the read from sample 1 passes."
            ),
        )

        data3, _ = read_rec_trodeChannels(
            rec, num_channels, [2], sampling_rate, header_size, 1, 100
        )
        np.testing.assert_allclose(
            np.asarray(data3).ravel(),
            expected["trode_channel_2_samples_1_100"],
            atol=1e-9,
            err_msg="Channel 2 differs: a stride error that drifts onto a neighbour.",
        )
