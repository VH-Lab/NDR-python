"""Generate symmetry artifacts for the VH Lab LabView (vld) reader.

Reads the checked-in example.vld/.vlh through ndr.reader.vld and exports JSON
artifacts the MATLAB symmetry suite re-reads and verifies.

Mirrors tools/tests/+ndr/+symmetry/+makeArtifacts/+reader/readVld.m
"""

import json
import shutil

import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, PYTHON_ARTIFACTS, json_safe

ARTIFACT_DIR = PYTHON_ARTIFACTS / "reader" / "readVld" / "testReadVldArtifacts"

S0, S1 = 1, 100


class TestReadVld:
    @pytest.fixture(autouse=True)
    def _setup(self):
        vld_file = EXAMPLE_DATA / "example.vld"
        if not vld_file.exists():
            pytest.skip("Example VLD file not available")

        from ndr.reader.vld import ndr_reader_vld

        self.reader = ndr_reader_vld()
        self.epochstreams = [str(vld_file)]

    def test_read_vld_artifacts(self):
        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        channels = self.reader.getchannelsepoch(self.epochstreams, 1)
        sr = self.reader.samplerate(self.epochstreams, 1, "ai", 1)
        t0t1 = self.reader.t0_t1(self.epochstreams, 1)
        ec = self.reader.epochclock(self.epochstreams, 1)

        metadata = {
            "channel_names": [c["name"] for c in channels],
            "channel_types": [c["type"] for c in channels],
            "samplerate": sr,
            "t0_t1": t0t1[0],
            "epochclock": [c.type for c in ec],
            "labeling_convention": self.reader.channelLabelingConvention("analog_in"),
        }
        (ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(json_safe(metadata), indent=2), encoding="utf-8"
        )

        # Channels 1 and 3 both land, so the artifact pins the channel->column
        # mapping and not merely the first column.
        ch1 = self.reader.readchannels_epochsamples("analog_in", [1], self.epochstreams, 1, S0, S1)
        ch3 = self.reader.readchannels_epochsamples("analog_in", [3], self.epochstreams, 1, S0, S1)
        time = self.reader.readchannels_epochsamples("time", [1], self.epochstreams, 1, S0, 10)

        read_struct = {
            "ai_channel_1_samples_1_100": ch1.ravel(),
            "ai_channel_3_samples_1_100": ch3.ravel(),
            "time_samples_1_10": time.ravel(),
        }
        (ARTIFACT_DIR / "readData.json").write_text(
            json.dumps(json_safe(read_struct), indent=2), encoding="utf-8"
        )
