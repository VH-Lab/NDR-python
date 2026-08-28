"""Read and verify symmetry artifacts for the tiffstack and prairieview readers.

Parameterized over both artifact sources, so one test verifies parity in both
directions. Skips when the artifact directory for a source does not exist.

Mirrors:
  tools/tests/+ndr/+symmetry/+readArtifacts/+reader/readTiffstack.m
  tools/tests/+ndr/+symmetry/+readArtifacts/+reader/readPrairieview.m
"""

import json

import numpy as np
import pytest

from tests.symmetry.conftest import EXAMPLE_DATA, SOURCE_TYPES, SYMMETRY_BASE


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestReadTiffstack:
    def _artifact_dir(self, source_type):
        return (
            SYMMETRY_BASE / source_type / "reader" / "readTiffstack" / "testReadTiffstackArtifacts"
        )

    def _reader(self):
        movie = EXAMPLE_DATA / "example_movie.tif"
        if not movie.exists():
            pytest.skip("Example TIFF movie not available")
        pytest.importorskip("tifffile")

        from ndr.reader.tiffstack import ndr_reader_tiffstack

        return ndr_reader_tiffstack(), [str(movie)]

    def test_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "metadata.json").read_text())
        reader, es = self._reader()

        assert reader.numframes(es, 1) == expected["numframes"]
        assert list(reader.framesize(es, 1)) == list(expected["framesize"])
        assert reader.dimensionorder(es, 1) == expected["dimensionorder"]
        # Strict string parity: both ports report the MATLAB class name.
        assert reader.datatype(es, 1) == expected["datatype"]
        assert reader.hasframetimes(es) == expected["hasframetimes"]

        np.testing.assert_allclose(reader.frametimes(es, 1), expected["frametimes"], atol=1e-9)
        np.testing.assert_allclose(reader.t0_t1(es, 1)[0], expected["t0_t1"], atol=1e-9)
        assert [c.type for c in reader.epochclock(es, 1)] == expected["epochclock"]

    def test_frames(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "readFrames.json").read_text())
        reader, es = self._reader()

        frames = reader.readframes(es, 1, [1, 5])
        np.testing.assert_array_equal(frames[:, :, 0, 0, 0], np.array(expected["frame_1"]))
        np.testing.assert_array_equal(frames[:, :, 0, 0, 1], np.array(expected["frame_5"]))


class TestReadPrairieview:
    def _artifact_dir(self, source_type):
        return (
            SYMMETRY_BASE
            / source_type
            / "reader"
            / "readPrairieview"
            / "testReadPrairieviewArtifacts"
        )

    def _reader(self):
        pv_dir = EXAMPLE_DATA / "prairieview"
        if not pv_dir.is_dir():
            pytest.skip("Example Prairie View directory not available")
        pytest.importorskip("tifffile")

        from ndr.reader.prairieview import ndr_reader_prairieview

        return ndr_reader_prairieview(), [str(pv_dir)]

    def test_metadata(self, source_type):
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "metadata.json").read_text())
        reader, es = self._reader()

        assert reader.numframes(es, 1) == expected["numframes"]
        assert list(reader.framesize(es, 1)) == list(expected["framesize"])
        assert reader.dimensionorder(es, 1) == expected["dimensionorder"]
        assert reader.datatype(es, 1) == expected["datatype"]
        assert reader.hasconfigtimes(es) == expected["hasconfigtimes"]

        np.testing.assert_allclose(reader.frametimes(es, 1), expected["frametimes"], atol=1e-9)
        np.testing.assert_allclose(reader.t0_t1(es, 1)[0], expected["t0_t1"], atol=1e-9)
        assert [c.type for c in reader.epochclock(es, 1)] == expected["epochclock"]

    def test_image_metadata(self, source_type):
        """The raster-scan metadata struct, field by field."""
        artifact_dir = self._artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        expected = json.loads((artifact_dir / "metadata.json").read_text())["image_metadata"]
        reader, es = self._reader()
        actual = reader.metadata(es, 1)

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
        reader, es = self._reader()

        frames = reader.readframes(es, 1, [1])
        np.testing.assert_array_equal(
            frames[:, :, 0, 0, 0], np.array(expected["frame_1_channel_1"])
        )
        np.testing.assert_array_equal(
            frames[:, :, 1, 0, 0], np.array(expected["frame_1_channel_2"])
        )

        sel = reader.readframes(es, 1, [2], SelectC=[2])
        np.testing.assert_array_equal(sel[:, :, 0, 0, 0], np.array(expected["frame_2_selectC_2"]))
