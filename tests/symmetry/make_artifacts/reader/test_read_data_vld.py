"""Generate python-side symmetry artifacts for the vld (VHLAB LabView) reader.

Reads example_data/vld_example.vld with the NDR-python vld reader and exports channel
metadata + a data sample to pythonArtifacts/, mirroring the MATLAB makeArtifacts step
(+ndr/+symmetry/+makeArtifacts/+reader/readData_vld.m). The read_artifacts test then
asserts the python reader reproduces BOTH the python and the matlab artifacts.
"""

import json
import shutil
from pathlib import Path

import pytest

from tests.symmetry.conftest import PYTHON_ARTIFACTS

ARTIFACT_DIR = PYTHON_ARTIFACTS / "reader" / "readData" / "testReadDataVldArtifacts"
EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"
VLD = EXAMPLE_DATA / "vld_example.vld"


class TestMakeVld:
    @pytest.fixture(autouse=True)
    def _setup(self):
        if not VLD.exists():
            pytest.skip("example_data/vld_example.vld not available")
        from ndr.reader.vld import ndr_reader_vld

        self.reader = ndr_reader_vld()
        self.epochfiles = [str(VLD)]

    def test_make_vld_artifacts(self):
        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        channels = self.reader.getchannelsepoch(self.epochfiles)
        sr = float(self.reader.samplerate(self.epochfiles, 1, "ai", 1))
        t0t1 = self.reader.t0_t1(self.epochfiles)
        ec = self.reader.epochclock(self.epochfiles)
        metadata = {
            "n_channels": len(channels),
            "channel_names": [c["name"] for c in channels],
            "samplerate": sr,
            "t0_t1": t0t1[0],
            "epochclock": [str(c) for c in ec],
        }
        (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        # ai1, samples 1..100 (1-based inclusive), in volts
        data = self.reader.readchannels_epochsamples("ai", [1], self.epochfiles, 1, 1, 100)
        (ARTIFACT_DIR / "readData.json").write_text(
            json.dumps({"ai_channel_1_samples_1_100": data.flatten().tolist()}, indent=2)
        )
