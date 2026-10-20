"""Generate symmetry artifacts for the video reader.

Exercises the frame API (numframes, framesize, dimensionorder, datatype,
frametimes, t0_t1, epochclock, getchannelsepoch, metadata, readframes) against
the checked-in example_video.avi (24-bit RGB) and example_video_grey.avi
(8-bit, grey palette) and exports JSON artifacts the MATLAB symmetry suite
re-reads and verifies. Both files are uncompressed, so the two ports must
agree pixel for pixel even though they decode with different libraries
(MATLAB's VideoReader, PyAV/FFmpeg here).

Each colour channel is stored as its own YxX plane (nested row lists, which is
what MATLAB's jsonencode produces for a matrix), so the comparison is
unambiguous despite MATLAB being column-major and numpy row-major.

Mirrors:
  tools/tests/+ndr/+symmetry/+makeArtifacts/+reader/readVideo.m
"""

import json
import shutil

import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, PYTHON_ARTIFACTS, json_safe

ARTIFACT_DIR = PYTHON_ARTIFACTS / "reader" / "readVideo" / "testReadVideoArtifacts"

VIDEOS = {"rgb": "example_video.avi", "grey": "example_video_grey.avi"}


def _fresh(d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


class TestReadVideo:
    @pytest.fixture(autouse=True)
    def _setup(self):
        for name in VIDEOS.values():
            if not (EXAMPLE_DATA / name).exists():
                pytest.skip(f"Example video {name} not available")
        pytest.importorskip("av")

        from ndr.reader.video import ndr_reader_video

        self.reader = ndr_reader_video()
        self.epochstreams = {k: [str(EXAMPLE_DATA / v)] for k, v in VIDEOS.items()}

    def _metadata(self, es):
        r = self.reader
        return {
            "numframes": r.numframes(es, 1),
            "framesize": r.framesize(es, 1),
            "dimensionorder": r.dimensionorder(es, 1),
            "datatype": r.datatype(es, 1),
            "frametimes": r.frametimes(es, 1),
            "t0_t1": r.t0_t1(es, 1)[0],
            "epochclock": [c.type for c in r.epochclock(es, 1)],
            "channels": [{"name": c["name"], "type": c["type"]} for c in r.getchannelsepoch(es, 1)],
            "image_metadata": r.metadata(es, 1),
        }

    def test_read_video_artifacts(self):
        _fresh(ARTIFACT_DIR)

        metadata = {k: self._metadata(es) for k, es in self.epochstreams.items()}
        (ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(json_safe(metadata), indent=2), encoding="utf-8"
        )

        rgb, grey = self.epochstreams["rgb"], self.epochstreams["grey"]
        n = metadata["rgb"]["numframes"]
        # First and last RGB frame, channel by channel; a SelectC read; and an
        # out-of-order greyscale read, which goes through the random-access path.
        frames = self.reader.readframes(rgb, 1, [1, n])
        sel = self.reader.readframes(rgb, 1, [3], SelectC=[2])
        greyframes = self.reader.readframes(grey, 1, [4, 2])
        read_struct = {}
        for t, frame in ((0, 1), (1, n)):
            for c in range(3):
                read_struct[f"rgb_frame_{frame}_channel_{c + 1}"] = frames[:, :, c, 0, t]
        read_struct["rgb_frame_3_selectC_2"] = sel[:, :, 0, 0, 0]
        read_struct["grey_frame_4"] = greyframes[:, :, 0, 0, 0]
        read_struct["grey_frame_2"] = greyframes[:, :, 0, 0, 1]
        (ARTIFACT_DIR / "readFrames.json").write_text(
            json.dumps(json_safe(read_struct), indent=2), encoding="utf-8"
        )
