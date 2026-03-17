"""Read and verify symmetry artifacts for NDR reader tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.symmetry.conftest import SOURCE_TYPES, SYMMETRY_BASE

EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadData:
    def _artifact_dir(self, source_type):
        return SYMMETRY_BASE / source_type / "reader" / "readData" / "testReadDataArtifacts"

    def test_read_data_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        metadata = json.loads((artifact_dir / "metadata.json").read_text())

        rhd_file = EXAMPLE_DATA / "Intan_160317_125049_short.rhd"
        if not rhd_file.exists():
            pytest.skip("Example RHD file not available")

        from ndr.reader.intan_rhd import IntanRHD

        reader = IntanRHD()
        epochfiles = [str(rhd_file)]

        actual_sr = reader.samplerate(epochfiles, 1, "ai", 1)
        assert (
            actual_sr == metadata["samplerate"]
        ), f"Sample rate mismatch: {actual_sr} vs {metadata['samplerate']}"

        actual_t0_t1 = reader.t0_t1(epochfiles)
        assert np.allclose(actual_t0_t1[0], metadata["t0_t1"], atol=1e-6)

    def test_read_data_samples(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        read_data = json.loads((artifact_dir / "readData.json").read_text())
        expected = np.array(read_data["ai_channel_1_samples_1_100"])

        rhd_file = EXAMPLE_DATA / "Intan_160317_125049_short.rhd"
        if not rhd_file.exists():
            pytest.skip("Example RHD file not available")

        from ndr.reader.intan_rhd import IntanRHD

        reader = IntanRHD()
        actual = reader.readchannels_epochsamples("ai", [1], [str(rhd_file)], 1, 1, 100)

        assert np.allclose(
            actual.flatten(), expected, atol=1e-9
        ), f"Data mismatch for ai channel 1, samples 1-100 ({source_type})"
