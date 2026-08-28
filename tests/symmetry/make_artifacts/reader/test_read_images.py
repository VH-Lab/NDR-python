"""Generate symmetry artifacts for the tiffstack and prairieview image readers.

Exercises the frame API (numframes, framesize, dimensionorder, datatype,
frametimes, readframes, metadata) against the checked-in example data and
exports JSON artifacts the MATLAB symmetry suite re-reads and verifies.

Image planes are stored as nested row lists, which is what MATLAB's jsonencode
produces for a matrix, so the comparison is unambiguous despite MATLAB being
column-major and numpy row-major.

Mirrors:
  tools/tests/+ndr/+symmetry/+makeArtifacts/+reader/readTiffstack.m
  tools/tests/+ndr/+symmetry/+makeArtifacts/+reader/readPrairieview.m
"""

import json
import shutil

import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, PYTHON_ARTIFACTS, json_safe

TIFF_ARTIFACT_DIR = PYTHON_ARTIFACTS / "reader" / "readTiffstack" / "testReadTiffstackArtifacts"
PV_ARTIFACT_DIR = PYTHON_ARTIFACTS / "reader" / "readPrairieview" / "testReadPrairieviewArtifacts"


def _fresh(d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


class TestReadTiffstack:
    @pytest.fixture(autouse=True)
    def _setup(self):
        movie = EXAMPLE_DATA / "example_movie.tif"
        if not movie.exists():
            pytest.skip("Example TIFF movie not available")
        pytest.importorskip("tifffile")

        from ndr.reader.tiffstack import ndr_reader_tiffstack

        self.reader = ndr_reader_tiffstack()
        self.epochstreams = [str(movie)]

    def test_read_tiffstack_artifacts(self):
        _fresh(TIFF_ARTIFACT_DIR)

        ec = self.reader.epochclock(self.epochstreams, 1)
        metadata = {
            "numframes": self.reader.numframes(self.epochstreams, 1),
            "framesize": self.reader.framesize(self.epochstreams, 1),
            "dimensionorder": self.reader.dimensionorder(self.epochstreams, 1),
            "datatype": self.reader.datatype(self.epochstreams, 1),
            "frametimes": self.reader.frametimes(self.epochstreams, 1),
            "t0_t1": self.reader.t0_t1(self.epochstreams, 1)[0],
            "epochclock": [c.type for c in ec],
            "hasframetimes": self.reader.hasframetimes(self.epochstreams),
        }
        (TIFF_ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(json_safe(metadata), indent=2), encoding="utf-8"
        )

        # First and last frame, as [Y][X] row lists.
        frames = self.reader.readframes(self.epochstreams, 1, [1, 5])
        read_struct = {
            "frame_1": frames[:, :, 0, 0, 0],
            "frame_5": frames[:, :, 0, 0, 1],
        }
        (TIFF_ARTIFACT_DIR / "readFrames.json").write_text(
            json.dumps(json_safe(read_struct), indent=2), encoding="utf-8"
        )


class TestReadPrairieview:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pv_dir = EXAMPLE_DATA / "prairieview"
        if not pv_dir.is_dir():
            pytest.skip("Example Prairie View directory not available")
        pytest.importorskip("tifffile")

        from ndr.reader.prairieview import ndr_reader_prairieview

        self.reader = ndr_reader_prairieview()
        self.epochstreams = [str(pv_dir)]

    def test_read_prairieview_artifacts(self):
        _fresh(PV_ARTIFACT_DIR)

        ec = self.reader.epochclock(self.epochstreams, 1)
        metadata = {
            "numframes": self.reader.numframes(self.epochstreams, 1),
            "framesize": self.reader.framesize(self.epochstreams, 1),
            "dimensionorder": self.reader.dimensionorder(self.epochstreams, 1),
            "datatype": self.reader.datatype(self.epochstreams, 1),
            "frametimes": self.reader.frametimes(self.epochstreams, 1),
            "t0_t1": self.reader.t0_t1(self.epochstreams, 1)[0],
            "epochclock": [c.type for c in ec],
            "hasconfigtimes": self.reader.hasconfigtimes(self.epochstreams),
            "image_metadata": self.reader.metadata(self.epochstreams, 1),
        }
        (PV_ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(json_safe(metadata), indent=2), encoding="utf-8"
        )

        # Both channels of frame 1, plus a SelectC-restricted read of frame 2,
        # which pins the C-axis ordering and the channel selection path.
        frames = self.reader.readframes(self.epochstreams, 1, [1])
        sel = self.reader.readframes(self.epochstreams, 1, [2], SelectC=[2])
        read_struct = {
            "frame_1_channel_1": frames[:, :, 0, 0, 0],
            "frame_1_channel_2": frames[:, :, 1, 0, 0],
            "frame_2_selectC_2": sel[:, :, 0, 0, 0],
        }
        (PV_ARTIFACT_DIR / "readFrames.json").write_text(
            json.dumps(json_safe(read_struct), indent=2), encoding="utf-8"
        )
