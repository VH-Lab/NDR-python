"""Generate python-side symmetry artifacts for the prairieview (Prairie View 2p) reader.

Reads example_data/prairieview_example/ (a legacy Prairie recording: one TIFF per
frame + a '*_Main.pcf' config) with the NDR-python prairieview reader and exports
frame metadata + a frame-1 pixel sample to pythonArtifacts/, mirroring the MATLAB
makeArtifacts step (+ndr/+symmetry/+makeArtifacts/+reader/readData_prairieview.m).
The read_artifacts test then asserts the python reader reproduces BOTH the python
and the matlab artifacts.

prairieview is a FRAME/IMAGE reader: it exposes numframes/framesize/datatype/
dimensionorder/frametimes/t0_t1/readframes (NOT channels/samplerate).
"""

import json
import shutil
from pathlib import Path

import pytest

from tests.symmetry.conftest import PYTHON_ARTIFACTS

ARTIFACT_DIR = PYTHON_ARTIFACTS / "reader" / "readData" / "testReadDataPrairieviewArtifacts"
EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"
PV_DIR = EXAMPLE_DATA / "prairieview_example"


class TestMakePrairieview:
    @pytest.fixture(autouse=True)
    def _setup(self):
        if not PV_DIR.is_dir():
            pytest.skip("example_data/prairieview_example/ not available")
        from ndr.reader.prairieview import ndr_reader_prairieview

        self.reader = ndr_reader_prairieview()
        self.epochfiles = [str(PV_DIR)]

    def test_make_prairieview_artifacts(self):
        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        numframes = int(self.reader.numframes(self.epochfiles, 1))
        framesize = [int(x) for x in self.reader.framesize(self.epochfiles, 1)]
        datatype = str(self.reader.datatype(self.epochfiles, 1))
        dimensionorder = str(self.reader.dimensionorder(self.epochfiles, 1))
        frametimes = [float(x) for x in self.reader.frametimes(self.epochfiles, 1).ravel()]
        t0t1 = [float(x) for x in self.reader.t0_t1(self.epochfiles, 1)[0]]
        ec = self.reader.epochclock(self.epochfiles, 1)
        metadata = {
            "numframes": numframes,
            "framesize": framesize,
            "datatype": datatype,
            "dimensionorder": dimensionorder,
            "frametimes": frametimes,
            "t0_t1": t0t1,
            "epochclock": [str(c) for c in ec],
        }
        (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        # frame 1 (1-based), all channels, pixels flattened
        frame1 = self.reader.readframes(self.epochfiles, 1, 1)
        (ARTIFACT_DIR / "readData.json").write_text(
            json.dumps({"frame_1_pixels": frame1.flatten().tolist()}, indent=2)
        )
