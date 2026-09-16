"""Tests for :class:`ndr.reader.smartspim.ndr_reader_smartspim`.

Python mirror of
``tools/tests/+ndr/+unittest/+reader/+smartspim/TestSmartSPIMReader.m``.

Metadata coverage: resolveepoch (pinned / unpinned channel / unpinned
tile / with-sidecars), numframes, framesize, dimensionorder, datatype,
epochclock, t0_t1, frametimes, getchannelsepoch.

Pixel-read coverage: single-frame read against the synthetic fixture,
multi-frame read in the requested order, different-channel epoch returns
different pixels (pinning actually routes), out-of-range frame indices
error.
"""

from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from ndr.reader.smartspim import ndr_reader_smartspim  # noqa: E402
from ndr.time.clocktype import ClockType  # noqa: E402
from tests._smartspim_fixture import make_example_fixture  # noqa: E402

_CH561 = "Ex_561_Em_561F_Ch2"
_CH640 = "Ex_640_Em_640F_Ch3"
_TILE1 = "100000/100000_200000"
_TILE2 = "100000/100000_300000"


@pytest.fixture
def fixture_dir(tmp_path):
    return make_example_fixture(tmp_path)


class TestResolveEpoch:
    def test_pinned_tile(self, fixture_dir):
        r = ndr_reader_smartspim()
        info = r.resolveepoch([str(fixture_dir), _CH561, _TILE1])
        assert info["rootDir"] == str(fixture_dir)
        assert info["channelName"] == _CH561
        assert info["tileId"] == _TILE1
        assert info["tile"]["numSlices"] == 4
        assert info["tile"]["height"] == 8
        assert info["tile"]["width"] == 10

    def test_unpinned_channel_errors(self, fixture_dir):
        r = ndr_reader_smartspim()
        with pytest.raises(ValueError, match="does not pin a channel"):
            r.resolveepoch(str(fixture_dir))
        with pytest.raises(ValueError, match="does not pin a channel"):
            r.resolveepoch([str(fixture_dir)])

    def test_unpinned_tile_errors(self, fixture_dir):
        r = ndr_reader_smartspim()
        with pytest.raises(ValueError, match="does not pin a tile"):
            r.resolveepoch([str(fixture_dir), _CH561])

    def test_missing_root_errors(self, fixture_dir, tmp_path):
        r = ndr_reader_smartspim()
        with pytest.raises((NotADirectoryError, FileNotFoundError)):
            r.resolveepoch([str(tmp_path / "does-not-exist"), _CH561, "x/y"])

    def test_sidecars_ignored(self, fixture_dir, tmp_path):
        sidecar = tmp_path / "sidecar.json"
        sidecar.write_text('{"note":"acquisition sidecar"}')
        r = ndr_reader_smartspim()
        info = r.resolveepoch([[str(fixture_dir), _CH640, _TILE2], str(sidecar)])
        assert info["channelName"] == _CH640
        assert info["tileId"] == _TILE2


class TestMetadata:
    def _es(self, fixture_dir):
        return [str(fixture_dir), _CH561, _TILE1]

    def test_numframes(self, fixture_dir):
        r = ndr_reader_smartspim()
        assert r.numframes(self._es(fixture_dir), 1) == 4

    def test_framesize(self, fixture_dir):
        r = ndr_reader_smartspim()
        assert r.framesize(self._es(fixture_dir), 1) == [8, 10, 1, 1, 4]

    def test_dimensionorder(self, fixture_dir):
        r = ndr_reader_smartspim()
        assert r.dimensionorder(self._es(fixture_dir), 1) == "YXCZT"

    def test_datatype(self, fixture_dir):
        r = ndr_reader_smartspim()
        assert r.datatype(self._es(fixture_dir), 1) == "uint16"

    def test_frametimes_all_nan(self, fixture_dir):
        r = ndr_reader_smartspim()
        t = r.frametimes(self._es(fixture_dir), 1, [1, 3])
        assert len(t) == 2
        assert np.all(np.isnan(t))

    def test_epochclock_is_no_time(self, fixture_dir):
        r = ndr_reader_smartspim()
        ec = r.epochclock(self._es(fixture_dir), 1)
        assert isinstance(ec, list)
        assert isinstance(ec[0], ClockType)
        assert ec[0].type == "no_time"

    def test_t0_t1_is_nan(self, fixture_dir):
        r = ndr_reader_smartspim()
        t0t1 = r.t0_t1(self._es(fixture_dir), 1)
        assert len(t0t1) == 1
        assert np.all(np.isnan(t0t1[0]))

    def test_getchannelsepoch(self, fixture_dir):
        r = ndr_reader_smartspim()
        ch = r.getchannelsepoch(self._es(fixture_dir), 1)
        assert len(ch) == 1
        assert ch[0]["name"] == "image1"
        assert ch[0]["type"] == "image"


class TestReadFrames:
    def test_single_frame_matches_fixture(self, fixture_dir):
        # Fixture rule: pixel_value(channelIndex=1, z=0) = 100
        r = ndr_reader_smartspim()
        frames = r.readframes([str(fixture_dir), _CH561, _TILE1], 1, [1])
        assert frames.shape == (8, 10, 1, 1, 1)
        assert frames.dtype == np.uint16
        assert frames[0, 0, 0, 0, 0] == np.uint16(100)

    def test_multiple_frames_in_order(self, fixture_dir):
        r = ndr_reader_smartspim()
        frame_inds = [1, 3, 4]
        frames = r.readframes([str(fixture_dir), _CH561, _TILE1], 1, frame_inds)
        assert frames.shape == (8, 10, 1, 1, 3)
        # Fixture: pixel = 100 * 1 + (frameIdx - 1)
        expected = np.array([100, 102, 103], dtype=np.uint16)
        actual = np.array(
            [
                frames[0, 0, 0, 0, 0],
                frames[0, 0, 0, 0, 1],
                frames[0, 0, 0, 0, 2],
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_different_channel_different_pixels(self, fixture_dir):
        # Pins actually route: Ex_640 has channelIndex=2, so pixel(z=0)=200.
        r = ndr_reader_smartspim()
        frames = r.readframes([str(fixture_dir), _CH640, _TILE1], 1, [1])
        assert frames[0, 0, 0, 0, 0] == np.uint16(200)

    def test_read_all_frames_default(self, fixture_dir):
        r = ndr_reader_smartspim()
        frames = r.readframes([str(fixture_dir), _CH561, _TILE1], 1, [])
        assert frames.shape == (8, 10, 1, 1, 4)

    def test_out_of_range_errors(self, fixture_dir):
        r = ndr_reader_smartspim()
        with pytest.raises((IndexError, ValueError)):
            r.readframes([str(fixture_dir), _CH561, _TILE1], 1, [1, 99])
