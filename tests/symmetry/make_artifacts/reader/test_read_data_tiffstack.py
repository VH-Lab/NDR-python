"""Generate python-side symmetry artifacts for the tiffstack (multipage TIFF) reader.

Reads example_data/tiffstack_example.tif with the NDR-python tiffstack frame/image
reader and exports frame metadata + frame-1 pixels to pythonArtifacts/, mirroring the
MATLAB makeArtifacts step (+ndr/+symmetry/+makeArtifacts/+reader/readData_tiffstack.m).
The read_artifacts test then asserts the python reader reproduces BOTH the python and
the matlab artifacts.

tiffstack is a frame/image reader, so the metadata is the frame API (numframes,
framesize, datatype, dimensionorder, epochclock, t0_t1), NOT channels/samplerate.
"""

import json
import shutil
from pathlib import Path

import pytest

from tests.symmetry.conftest import PYTHON_ARTIFACTS

ARTIFACT_DIR = (
    PYTHON_ARTIFACTS / "reader" / "readData" / "testReadDataTiffstackArtifacts"
)
EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"
TIFF = EXAMPLE_DATA / "tiffstack_example.tif"


class TestMakeTiffstack:
    @pytest.fixture(autouse=True)
    def _setup(self):
        if not TIFF.exists():
            pytest.skip("example_data/tiffstack_example.tif not available")
        from ndr.reader.tiffstack import ndr_reader_tiffstack

        self.reader = ndr_reader_tiffstack()
        self.epochfiles = [str(TIFF)]

    def test_make_tiffstack_artifacts(self):
        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        ef = self.epochfiles
        numframes = int(self.reader.numframes(ef, 1))
        framesize = [int(x) for x in self.reader.framesize(ef, 1)]
        datatype = self.reader.datatype(ef, 1)
        dimensionorder = self.reader.dimensionorder(ef, 1)
        ec = self.reader.epochclock(ef, 1)
        t0t1 = self.reader.t0_t1(ef, 1)
        metadata = {
            "numframes": numframes,
            "framesize": framesize,
            "datatype": datatype,
            "dimensionorder": dimensionorder,
            "epochclock": [str(c) for c in ec],
            "t0_t1": t0t1[0],
        }
        (ARTIFACT_DIR / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

        # frame 1 (1-based), flattened to a JSON list.
        frame1 = self.reader.readframes(ef, 1, [1])
        (ARTIFACT_DIR / "readData.json").write_text(
            json.dumps({"frame_1_pixels": frame1.flatten().tolist()}, indent=2)
        )
