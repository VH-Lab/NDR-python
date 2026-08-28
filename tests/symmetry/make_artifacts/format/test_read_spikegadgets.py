"""Generate symmetry artifacts for the SpikeGadgets format layer.

Compares at ndr.format.spikegadgets rather than through
ndr.reader.spikegadgets_rec, whose readchannels_epochsamples is still a stub;
the format functions are what actually read the file.

The artifacts are deliberately alignment-sensitive. A .rec is a stream of
fixed-size packets, so both the offset of the first packet and the stride
between packets must be exact. Get either wrong and the values stay in range
and the arrays keep their shape -- only a byte-for-byte comparison against the
other port notices.

Mirrors tools/tests/+ndr/+symmetry/+makeArtifacts/+format/readSpikeGadgets.m
"""

import json
import shutil

import numpy as np
import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, PYTHON_ARTIFACTS, json_safe

ARTIFACT_DIR = PYTHON_ARTIFACTS / "format" / "readSpikeGadgets" / "testReadSpikeGadgetsArtifacts"


class TestReadSpikeGadgets:
    @pytest.fixture(autouse=True)
    def _setup(self):
        rec = EXAMPLE_DATA / "example.rec"
        if not rec.exists():
            pytest.skip("Example .rec file not available")
        self.rec = rec

    def test_read_spikegadgets_artifacts(self):
        from ndr.format.spikegadgets.read_rec_config import read_rec_config
        from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels

        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        config, _channels = read_rec_config(str(self.rec))
        num_channels = int(config["numChannels"])
        header_size = int(config["headerSize"])
        sampling_rate = float(config["samplingRate"])

        metadata = {
            "numChannels": num_channels,
            "headerSize": header_size,
            "samplingRate": sampling_rate,
        }
        (ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(json_safe(metadata), indent=2), encoding="utf-8"
        )

        # Trode channel 1 from the start of the file.
        data1, ts1 = read_rec_trodeChannels(
            self.rec, num_channels, [1], sampling_rate, header_size, 1, 100
        )
        # The same channel from s0 = 1001: an error in the seek to s0 scales
        # with s0, so it does not show up in the first read.
        data2, _ = read_rec_trodeChannels(
            self.rec, num_channels, [1], sampling_rate, header_size, 1001, 1100
        )
        # A second channel, to catch a drift that lands on a neighbour.
        data3, _ = read_rec_trodeChannels(
            self.rec, num_channels, [2], sampling_rate, header_size, 1, 100
        )

        read_struct = {
            "trode_channel_1_samples_1_100": np.asarray(data1).ravel(),
            "trode_timestamps_samples_1_100": np.asarray(ts1).ravel(),
            "trode_channel_1_samples_1001_1100": np.asarray(data2).ravel(),
            "trode_channel_2_samples_1_100": np.asarray(data3).ravel(),
        }
        (ARTIFACT_DIR / "readData.json").write_text(
            json.dumps(json_safe(read_struct), indent=2), encoding="utf-8"
        )

        assert (ARTIFACT_DIR / "readData.json").exists()
