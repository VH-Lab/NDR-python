"""Tests for :class:`ndr.reader.omezarr.ndr_reader_omezarr`.

Python mirror of
``tools/tests/+ndr/+unittest/+reader/+omezarr/TestOMEZarrReader.m``.

Metadata coverage: resolveepoch (pinned / unpinned / unknown /
with-sidecars), numframes, framesize, dimensionorder, datatype,
epochclock, t0_t1, frametimes, getchannelsepoch.

Pixel-read coverage: single frame matches ground truth, multi-frame
in requested order, ``Level`` selects a lower-resolution level, and
mean vs max return different pixels at level 2 (the pyramid pinning
actually routes).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("zarr")

from ndr.reader.omezarr import ndr_reader_omezarr  # noqa: E402
from ndr.time.clocktype import ClockType  # noqa: E402
from tests._omezarr_fixture import make_example_fixture  # noqa: E402


@pytest.fixture
def fixture(tmp_path):
    return make_example_fixture(tmp_path, with_chunks=True)


def _z_slice(vol: np.ndarray, z_index: int) -> np.ndarray:
    """Extract one z-plane from a (c, z, y, x) volume as (y, x, c)."""
    plane_czyx = vol[:, z_index - 1, :, :]
    plane_yxc = np.transpose(plane_czyx, (1, 2, 0))
    return plane_yxc.reshape(plane_yxc.shape[0], plane_yxc.shape[1], plane_yxc.shape[2])


class TestResolveEpoch:
    def test_pinned_pyramid(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        info = r.resolveepoch([str(fixture_dir), "mean"])
        assert info["zarrPath"] == str(fixture_dir)
        assert info["pyramidName"] == "mean"
        assert info["axisIndex"]["c"] == 1
        assert info["axisIndex"]["z"] == 2
        assert info["axisIndex"]["y"] == 3
        assert info["axisIndex"]["x"] == 4

    def test_unpinned_pyramid_errors(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        with pytest.raises(ValueError, match="does not pin a pyramid"):
            r.resolveepoch(str(fixture_dir))
        with pytest.raises(ValueError, match="does not pin a pyramid"):
            r.resolveepoch([str(fixture_dir)])

    def test_unknown_pyramid_errors(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        with pytest.raises(ValueError, match='"median"'):
            r.resolveepoch([str(fixture_dir), "median"])

    def test_sidecars_ignored(self, fixture, tmp_path):
        fixture_dir, _ = fixture
        sidecar = tmp_path / "sidecar.json"
        sidecar.write_text('{"note":"acquisition sidecar"}')
        r = ndr_reader_omezarr()
        info = r.resolveepoch([[str(fixture_dir), "max"], str(sidecar)])
        assert info["pyramidName"] == "max"
        assert info["zarrPath"] == str(fixture_dir)


class TestMetadata:
    def test_numframes(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        assert r.numframes([str(fixture_dir), "mean"], 1) == 8

    def test_framesize(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        assert r.framesize([str(fixture_dir), "mean"], 1) == [8, 10, 1, 1, 8]

    def test_dimensionorder(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        assert r.dimensionorder([str(fixture_dir), "mean"], 1) == "YXCZT"

    def test_datatype(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        assert r.datatype([str(fixture_dir), "mean"], 1) == "uint16"

    def test_frametimes_all_nan(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        t = r.frametimes([str(fixture_dir), "mean"], 1, [1, 5, 8])
        assert len(t) == 3
        assert np.all(np.isnan(t))

    def test_epochclock_is_no_time(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        ec = r.epochclock([str(fixture_dir), "mean"], 1)
        assert isinstance(ec, list)
        assert isinstance(ec[0], ClockType)
        assert ec[0].type == "no_time"

    def test_t0_t1_is_nan(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        t0t1 = r.t0_t1([str(fixture_dir), "mean"], 1)
        assert len(t0t1) == 1
        assert np.all(np.isnan(t0t1[0]))

    def test_getchannelsepoch(self, fixture):
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        ch = r.getchannelsepoch([str(fixture_dir), "mean"], 1)
        assert len(ch) == 1
        assert ch[0]["name"] == "image1"
        assert ch[0]["type"] == "image"


class TestReadFrames:
    def test_single_frame_matches_ground_truth(self, fixture):
        fixture_dir, gt = fixture
        r = ndr_reader_omezarr()
        frame_idx = 3
        frames = r.readframes([str(fixture_dir), "mean"], 1, [frame_idx])
        expected = _z_slice(gt.level0, frame_idx)
        assert frames.shape == (8, 10, 1, 1, 1)
        assert frames.dtype == np.uint16
        np.testing.assert_array_equal(frames[:, :, :, 0, 0], expected)

    def test_multiple_frames_in_order(self, fixture):
        fixture_dir, gt = fixture
        r = ndr_reader_omezarr()
        frame_inds = [1, 4, 7]
        frames = r.readframes([str(fixture_dir), "mean"], 1, frame_inds)
        assert frames.shape == (8, 10, 1, 1, 3)
        for k, f in enumerate(frame_inds):
            expected = _z_slice(gt.level0, f)
            np.testing.assert_array_equal(frames[:, :, :, 0, k], expected)

    def test_level_option_selects_lower_resolution(self, fixture):
        # Level 2 in the reader is pyramid.levels[1], which points at
        # 'mean/1' -- the 2x downsample (fixture gt.mean_level1 shape
        # [1 4 4 5]). Level 1 is the shared root.
        fixture_dir, gt = fixture
        r = ndr_reader_omezarr()
        frames = r.readframes([str(fixture_dir), "mean"], 1, [1], Level=2)
        expected = _z_slice(gt.mean_level1, 1)
        assert frames.shape == (4, 5, 1, 1, 1)
        np.testing.assert_array_equal(frames[:, :, :, 0, 0], expected)

    def test_mean_and_max_differ_at_level2(self, fixture):
        # The whole point of pinning a pyramid: mean and max return
        # different pixels at the same coordinates.
        fixture_dir, gt = fixture
        assert not np.array_equal(gt.mean_level1, gt.max_level1)

        r = ndr_reader_omezarr()
        mean_frame = r.readframes([str(fixture_dir), "mean"], 1, [1], Level=2)
        max_frame = r.readframes([str(fixture_dir), "max"], 1, [1], Level=2)
        np.testing.assert_array_equal(mean_frame[:, :, :, 0, 0], _z_slice(gt.mean_level1, 1))
        np.testing.assert_array_equal(max_frame[:, :, :, 0, 0], _z_slice(gt.max_level1, 1))
        assert not np.array_equal(mean_frame, max_frame)

    def test_all_frames_default(self, fixture):
        # An empty frameind reads all frames.
        fixture_dir, _ = fixture
        r = ndr_reader_omezarr()
        frames = r.readframes([str(fixture_dir), "mean"], 1, [])
        assert frames.shape == (8, 10, 1, 1, 8)


class TestZarrClassMapping:
    def test_maps_common_dtypes(self):
        assert ndr_reader_omezarr.zarrClass("<u2") == "uint16"
        assert ndr_reader_omezarr.zarrClass(">i4") == "int32"
        assert ndr_reader_omezarr.zarrClass("|u1") == "uint8"
        assert ndr_reader_omezarr.zarrClass("f8") == "double"

    def test_rejects_unsupported(self):
        with pytest.raises(ValueError):
            ndr_reader_omezarr.zarrClass("<c16")
