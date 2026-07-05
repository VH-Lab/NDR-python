"""Symmetry verification for the tiffstack (multipage TIFF) frame/image reader.

For each artifact source (matlabArtifacts + pythonArtifacts), re-read the SAME
example_data/tiffstack_example.tif with the NDR-python tiffstack reader and assert its
output matches the stored artifacts. The matlabArtifacts are produced by the NDR-matlab
makeArtifacts step (+ndr/+symmetry/+makeArtifacts/+reader/readData_tiffstack.m); when
they are absent (python-only CI) that source is skipped — the pythonArtifacts source
always runs. This proves NDR-matlab and NDR-python read the .tif identically.

tiffstack is a frame/image reader, so the checked metadata is the frame API
(numframes/framesize/datatype) plus the flattened frame-1 pixels.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.symmetry.conftest import SOURCE_TYPES, SYMMETRY_BASE

EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"
TIFF = EXAMPLE_DATA / "tiffstack_example.tif"


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadDataTiffstack:
    def _artifact_dir(self, source_type):
        return (
            SYMMETRY_BASE
            / source_type
            / "reader"
            / "readData"
            / "testReadDataTiffstackArtifacts"
        )

    def _reader(self):
        from ndr.reader.tiffstack import ndr_reader_tiffstack

        return ndr_reader_tiffstack(), [str(TIFF)]

    def test_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")
        if not TIFF.exists():
            pytest.skip("example_data/tiffstack_example.tif not available")

        meta = json.loads((artifact_dir / "metadata.json").read_text())
        reader, ef = self._reader()

        assert int(reader.numframes(ef, 1)) == meta["numframes"]
        assert [int(x) for x in reader.framesize(ef, 1)] == list(meta["framesize"])
        assert reader.datatype(ef, 1) == meta["datatype"]

    def test_frame_pixels(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")
        if not TIFF.exists():
            pytest.skip("example_data/tiffstack_example.tif not available")

        expected = np.array(
            json.loads((artifact_dir / "readData.json").read_text())["frame_1_pixels"]
        )
        reader, ef = self._reader()
        actual = reader.readframes(ef, 1, [1]).flatten()
        assert np.array_equal(actual, expected), (
            f"tiffstack frame-1 pixel mismatch vs {source_type}"
        )
