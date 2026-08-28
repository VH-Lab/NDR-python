"""Read and verify symmetry artifacts for the VH Lab LabView (vld) reader.

Parameterized over both artifact sources, so one test verifies parity in both
directions. Skips when the artifact directory for a source does not exist.

Mirrors tools/tests/+ndr/+symmetry/+readArtifacts/+reader/readVld.m
"""

import json

import numpy as np
import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, SOURCE_TYPES, SYMMETRY_BASE

S0, S1 = 1, 100


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadVld:
    def _artifact_dir(self, source_type):
        return SYMMETRY_BASE / source_type / "reader" / "readVld" / "testReadVldArtifacts"

    def _reader(self):
        vld_file = EXAMPLE_DATA / "example.vld"
        if not vld_file.exists():
            pytest.skip("Example VLD file not available")

        from ndr.reader.vld import ndr_reader_vld

        return ndr_reader_vld(), [str(vld_file)]

    def test_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "metadata.json").read_text())
        reader, epochstreams = self._reader()

        channels = reader.getchannelsepoch(epochstreams, 1)
        assert [c["name"] for c in channels] == expected["channel_names"]
        assert [c["type"] for c in channels] == expected["channel_types"]

        assert reader.samplerate(epochstreams, 1, "ai", 1) == pytest.approx(
            expected["samplerate"], abs=1e-9
        )
        np.testing.assert_allclose(reader.t0_t1(epochstreams, 1)[0], expected["t0_t1"], atol=1e-6)
        assert [c.type for c in reader.epochclock(epochstreams, 1)] == expected["epochclock"]
        assert reader.channelLabelingConvention("analog_in") == expected["labeling_convention"]

    def test_samples(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "readData.json").read_text())
        reader, epochstreams = self._reader()

        ch1 = reader.readchannels_epochsamples("analog_in", [1], epochstreams, 1, S0, S1)
        np.testing.assert_allclose(ch1.ravel(), expected["ai_channel_1_samples_1_100"], atol=1e-9)

        ch3 = reader.readchannels_epochsamples("analog_in", [3], epochstreams, 1, S0, S1)
        np.testing.assert_allclose(ch3.ravel(), expected["ai_channel_3_samples_1_100"], atol=1e-9)

        time = reader.readchannels_epochsamples("time", [1], epochstreams, 1, S0, 10)
        np.testing.assert_allclose(time.ravel(), expected["time_samples_1_10"], atol=1e-9)
