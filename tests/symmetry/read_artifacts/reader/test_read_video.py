"""Read and verify symmetry artifacts for the video reader.

Parameterized over both artifact sources, so one test verifies parity in both
directions. Skips when the artifact directory for a source does not exist.

Frame times are compared to 1e-6 s rather than the 1e-9 the TIFF readers use:
here each port measures them through its own decoder (VideoReader's
CurrentTime in MATLAB, FFmpeg presentation timestamps here) rather than
reading the same numbers from a text sidecar, and a microsecond is still far
below one frame period. Pixels must match exactly: the files are uncompressed.

Mirrors:
  tools/tests/+ndr/+symmetry/+readArtifacts/+reader/readVideo.m
"""

import json

import numpy as np
import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, SOURCE_TYPES, SYMMETRY_BASE

VIDEOS = {"rgb": "example_video.avi", "grey": "example_video_grey.avi"}
TIME_TOL = 1e-6


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadVideo:
    def _artifact_dir(self, source_type):
        return SYMMETRY_BASE / source_type / "reader" / "readVideo" / "testReadVideoArtifacts"

    def _reader(self):
        for name in VIDEOS.values():
            if not (EXAMPLE_DATA / name).exists():
                pytest.skip(f"Example video {name} not available")
        pytest.importorskip("av")

        from ndr.reader.video import ndr_reader_video

        return ndr_reader_video(), {k: [str(EXAMPLE_DATA / v)] for k, v in VIDEOS.items()}

    @pytest.mark.parametrize("video", sorted(VIDEOS))
    def test_metadata(self, source_type, video):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "metadata.json").read_text())[video]
        reader, streams = self._reader()
        es = streams[video]

        assert reader.numframes(es, 1) == expected["numframes"]
        assert list(reader.framesize(es, 1)) == list(expected["framesize"])
        assert reader.dimensionorder(es, 1) == expected["dimensionorder"]
        # Strict string parity: both ports report the MATLAB class name.
        assert reader.datatype(es, 1) == expected["datatype"]
        np.testing.assert_allclose(
            reader.frametimes(es, 1), np.ravel(expected["frametimes"]), atol=TIME_TOL
        )
        np.testing.assert_allclose(reader.t0_t1(es, 1)[0], expected["t0_t1"], atol=TIME_TOL)
        assert [c.type for c in reader.epochclock(es, 1)] == list(expected["epochclock"])

        # MATLAB's jsondecode turns a one-element struct array into a struct.
        exp_channels = expected["channels"]
        if isinstance(exp_channels, dict):
            exp_channels = [exp_channels]
        assert [
            {"name": c["name"], "type": c["type"]} for c in reader.getchannelsepoch(es, 1)
        ] == exp_channels

    @pytest.mark.parametrize("video", sorted(VIDEOS))
    def test_image_metadata(self, source_type, video):
        """The image-metadata struct, field by field."""
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "metadata.json").read_text())[video]["image_metadata"]
        reader, streams = self._reader()
        actual = reader.metadata(streams[video], 1)

        assert set(actual) == set(expected), "image metadata field sets differ"
        for key, exp in expected.items():
            got = actual[key]
            if exp is None:
                # MATLAB writes NaN as null; the Python side must also be NaN.
                assert isinstance(got, float) and np.isnan(got), f"{key} should be NaN"
            elif isinstance(exp, bool):
                assert bool(got) == exp, f"{key} mismatch"
            else:
                assert got == pytest.approx(exp, abs=1e-9), f"{key} mismatch"

    def test_frames(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "readFrames.json").read_text())
        reader, streams = self._reader()
        rgb, grey = streams["rgb"], streams["grey"]

        n = reader.numframes(rgb, 1)
        frames = reader.readframes(rgb, 1, [1, n])
        for t, frame in ((0, 1), (1, n)):
            for c in range(3):
                key = f"rgb_frame_{frame}_channel_{c + 1}"
                np.testing.assert_array_equal(
                    frames[:, :, c, 0, t], np.array(expected[key]), err_msg=key
                )

        sel = reader.readframes(rgb, 1, [3], SelectC=[2])
        np.testing.assert_array_equal(
            sel[:, :, 0, 0, 0], np.array(expected["rgb_frame_3_selectC_2"])
        )

        greyframes = reader.readframes(grey, 1, [4, 2])
        np.testing.assert_array_equal(greyframes[:, :, 0, 0, 0], np.array(expected["grey_frame_4"]))
        np.testing.assert_array_equal(greyframes[:, :, 0, 0, 1], np.array(expected["grey_frame_2"]))
