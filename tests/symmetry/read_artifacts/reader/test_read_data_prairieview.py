"""Symmetry verification for the prairieview (Prairie View 2p) reader.

For each artifact source (matlabArtifacts + pythonArtifacts), re-read the SAME
example_data/prairieview_example/ recording with the NDR-python prairieview reader
and assert its output matches the stored artifacts. The matlabArtifacts are produced
by the NDR-matlab makeArtifacts step
(+ndr/+symmetry/+makeArtifacts/+reader/readData_prairieview.m); when they are absent
(python-only CI) that source is skipped — the pythonArtifacts source always runs.
This proves NDR-matlab and NDR-python read the Prairie recording identically.

prairieview is a FRAME/IMAGE reader: numframes/framesize/datatype/dimensionorder/
frametimes/t0_t1/readframes (NOT channels/samplerate).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.symmetry.conftest import SOURCE_TYPES, SYMMETRY_BASE

EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"
PV_DIR = EXAMPLE_DATA / "prairieview_example"


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadDataPrairieview:
    def _artifact_dir(self, source_type):
        return (
            SYMMETRY_BASE
            / source_type
            / "reader"
            / "readData"
            / "testReadDataPrairieviewArtifacts"
        )

    def _reader(self):
        from ndr.reader.prairieview import ndr_reader_prairieview

        return ndr_reader_prairieview(), [str(PV_DIR)]

    def test_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")
        if not PV_DIR.is_dir():
            pytest.skip("example_data/prairieview_example/ not available")

        meta = json.loads((artifact_dir / "metadata.json").read_text())
        reader, ef = self._reader()

        assert int(reader.numframes(ef, 1)) == meta["numframes"]
        assert [int(x) for x in reader.framesize(ef, 1)] == meta["framesize"]
        assert str(reader.datatype(ef, 1)) == meta["datatype"]
        assert str(reader.dimensionorder(ef, 1)) == meta["dimensionorder"]
        assert np.allclose(
            reader.frametimes(ef, 1).ravel(), meta["frametimes"], atol=1e-9
        )
        assert np.allclose(reader.t0_t1(ef, 1)[0], meta["t0_t1"], atol=1e-9)

    def test_frame_pixels(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")
        if not PV_DIR.is_dir():
            pytest.skip("example_data/prairieview_example/ not available")

        expected = np.array(
            json.loads((artifact_dir / "readData.json").read_text())["frame_1_pixels"]
        )
        reader, ef = self._reader()
        actual = reader.readframes(ef, 1, 1).flatten()
        assert np.array_equal(actual, expected), (
            f"prairieview frame-1 pixel mismatch vs {source_type}"
        )
