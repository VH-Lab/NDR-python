"""Generate symmetry artifacts for NDR reader tests.

Reads example data files using NDR readers and exports:
- Channel metadata (names, types, sample rates)
- Epoch clock types and t0/t1 boundaries
- Actual data samples for comparison
"""

import json
import shutil

import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, PYTHON_ARTIFACTS

ARTIFACT_DIR = PYTHON_ARTIFACTS / "reader" / "readData" / "testReadDataArtifacts"


class TestReadData:
    @pytest.fixture(autouse=True)
    def _setup(self):
        rhd_file = EXAMPLE_DATA / "example.rhd"
        if not rhd_file.exists():
            pytest.skip("Example RHD file not available")

        from ndr.reader.intan_rhd import ndr_reader_intan__rhd

        self.reader = ndr_reader_intan__rhd()
        self.epochfiles = [str(rhd_file)]

    def test_read_data_artifacts(self):
        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        # Export channel metadata
        channels = self.reader.getchannelsepoch(self.epochfiles)
        sr = self.reader.samplerate(self.epochfiles, 1, "ai", 1)
        t0t1 = self.reader.t0_t1(self.epochfiles)
        ec = self.reader.epochclock(self.epochfiles)

        metadata = {
            "channels": channels,
            "samplerate": sr,
            "t0_t1": t0t1[0],
            "epochclock": [str(c) for c in ec],
        }
        (ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        # Export a small data sample
        data = self.reader.readchannels_epochsamples("ai", [1], self.epochfiles, 1, 1, 100)
        (ARTIFACT_DIR / "readData.json").write_text(
            json.dumps({"ai_channel_1_samples_1_100": data.flatten().tolist()}, indent=2),
            encoding="utf-8",
        )
