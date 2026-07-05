"""Symmetry verification for the vld (VHLAB LabView) reader.

For each artifact source (matlabArtifacts + pythonArtifacts), re-read the SAME
example_data/vld_example.vld with the NDR-python vld reader and assert its output
matches the stored artifacts byte-for-byte. The matlabArtifacts are produced by the
NDR-matlab makeArtifacts step (+ndr/+symmetry/+makeArtifacts/+reader/readData_vld.m);
when they are absent (python-only CI) that source is skipped — the pythonArtifacts
source always runs. This proves NDR-matlab and NDR-python read the .vld identically.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.symmetry.conftest import SOURCE_TYPES, SYMMETRY_BASE

EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"
VLD = EXAMPLE_DATA / "vld_example.vld"


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadDataVld:
    def _artifact_dir(self, source_type):
        return SYMMETRY_BASE / source_type / "reader" / "readData" / "testReadDataVldArtifacts"

    def _reader(self):
        from ndr.reader.vld import ndr_reader_vld

        return ndr_reader_vld(), [str(VLD)]

    def test_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")
        if not VLD.exists():
            pytest.skip("example_data/vld_example.vld not available")

        meta = json.loads((artifact_dir / "metadata.json").read_text())
        reader, ef = self._reader()

        assert float(reader.samplerate(ef, 1, "ai", 1)) == meta["samplerate"]
        assert np.allclose(reader.t0_t1(ef)[0], meta["t0_t1"], atol=1e-9)
        channels = reader.getchannelsepoch(ef)
        assert len(channels) == meta["n_channels"]

    def test_samples(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")
        if not VLD.exists():
            pytest.skip("example_data/vld_example.vld not available")

        expected = np.array(json.loads((artifact_dir / "readData.json").read_text())["ai_channel_1_samples_1_100"])
        reader, ef = self._reader()
        actual = reader.readchannels_epochsamples("ai", [1], ef, 1, 1, 100).flatten()
        assert np.allclose(actual, expected, atol=1e-9), f"vld sample mismatch vs {source_type}"
