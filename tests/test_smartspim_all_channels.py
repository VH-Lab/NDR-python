"""Tests for the SmartSPIM ``All_Channels`` (stitched volume) support.

Python mirror of
``tools/tests/+ndr/+unittest/+format/+smartspim/TestSmartSPIMAllChannels.m``
and
``tools/tests/+ndr/+unittest/+reader/+smartspim/TestSmartSPIMReaderAllChannels.m``.

Fixture: the usual 2-channel, 2-tile acquisition plus an All_Channels
folder of 4 x 106x13 uint16 LZW slices per channel with pixel value
``1000 * (k + 1) + z`` for file ``_ch<k>``, slice ``z`` (0-based).
Without ``_MIP_stitched`` folders, Ex_561_Em_561F_Ch2 -> ch0 and
Ex_640_Em_640F_Ch3 -> ch1 (filter-channel rank).
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("imagecodecs")  # LZW, as in real All_Channels slices

from ndr.format.smartspim import (  # noqa: E402
    hasAllChannels,
    readAllChannelsInfo,
    readAllChannelsVolume,
)
from ndr.reader.smartspim import ndr_reader_smartspim  # noqa: E402
from tests._smartspim_fixture import make_example_fixture  # noqa: E402

_CH561 = "Ex_561_Em_561F_Ch2"
_CH640 = "Ex_640_Em_640F_Ch3"
_TILE1 = "100000/100000_200000"


@pytest.fixture
def fixture_dir(tmp_path):
    return make_example_fixture(tmp_path, all_channels=True)


def _all_dir(fixture_dir: Path) -> Path:
    return fixture_dir / "All_Channels"


# ----------------------------------------------------------------------
# Format layer
# ----------------------------------------------------------------------


class TestHasAllChannels:
    def test_true(self, fixture_dir):
        assert hasAllChannels(str(fixture_dir)) is True

    def test_false_without_folder(self, fixture_dir):
        shutil.rmtree(_all_dir(fixture_dir))
        assert hasAllChannels(str(fixture_dir)) is False

    def test_false_for_bad_input(self, tmp_path):
        assert hasAllChannels(str(tmp_path / "nope")) is False
        assert hasAllChannels(42) is False

    def test_false_when_channel_missing(self, fixture_dir):
        for p in _all_dir(fixture_dir).glob("*_ch1.tiff"):
            p.unlink()
        assert hasAllChannels(str(fixture_dir)) is False
        with pytest.raises(FileNotFoundError, match="_ch1"):
            readAllChannelsInfo(str(fixture_dir))


class TestReadAllChannelsInfo:
    def test_channels_by_filter_channel_rank(self, fixture_dir):
        info = readAllChannelsInfo(str(fixture_dir))
        assert info["channelNames"] == [_CH561, _CH640]
        assert [c["chIndex"] for c in info["channels"]] == [0, 1]
        assert [c["numSlices"] for c in info["channels"]] == [4, 4]
        assert info["height"] == 106
        assert info["width"] == 13
        assert info["dtype"] == "uint16"
        assert info["voxelSizeUm"] == [4, 4, 4]
        assert Path(info["channels"][1]["slicePaths"][-1]).name == "200000_100000_000030_ch1.tiff"

    def test_without_header(self, fixture_dir):
        info = readAllChannelsInfo(str(fixture_dir), ReadHeader=False)
        assert math.isnan(info["height"])
        assert info["dtype"] == ""

    def test_slices_sorted_numerically(self, fixture_dir):
        # "5" sorts after "000030" as text but between 0 and 10 by value.
        for k in range(2):
            img = np.full((106, 13), 1000 * (k + 1) + 99, dtype=np.uint16)
            tifffile.imwrite(_all_dir(fixture_dir) / f"200000_100000_5_ch{k}.tiff", img)
        info = readAllChannelsInfo(str(fixture_dir))
        assert info["channels"][0]["numSlices"] == 5
        assert Path(info["channels"][0]["slicePaths"][1]).stem == "200000_100000_5_ch0"

    def test_mip_stitched_overrides_rank(self, fixture_dir):
        for name, k in ((_CH561, 1), (_CH640, 0)):
            mip = fixture_dir / f"{name}_MIP_stitched"
            mip.mkdir()
            tifffile.imwrite(mip / f"200000_100000_000000_ch{k}.tiff", np.zeros((2, 2), np.uint16))
        info = readAllChannelsInfo(str(fixture_dir))
        assert info["channelNames"] == [_CH640, _CH561]
        vol = readAllChannelsVolume(str(fixture_dir), _CH561, ZRange=[1, 1])
        assert vol[0, 0, 0] == 2000

    def test_errors_without_folder(self, fixture_dir):
        shutil.rmtree(_all_dir(fixture_dir))
        with pytest.raises(FileNotFoundError, match="All_Channels directory not found"):
            readAllChannelsInfo(str(fixture_dir))

    def test_errors_without_slices(self, fixture_dir):
        for p in _all_dir(fixture_dir).glob("*.tiff"):
            p.unlink()
        with pytest.raises(FileNotFoundError, match="No"):
            readAllChannelsInfo(str(fixture_dir))


class TestReadAllChannelsVolume:
    def test_full(self, fixture_dir):
        vol = readAllChannelsVolume(str(fixture_dir), _CH640)
        assert vol.shape == (4, 106, 13)
        assert vol.dtype == np.uint16
        assert vol[:, 4, 6].tolist() == [2000, 2001, 2002, 2003]

    def test_zrange(self, fixture_dir):
        vol = readAllChannelsVolume(str(fixture_dir), _CH561, ZRange=[2, 3])
        assert vol.shape == (2, 106, 13)
        assert vol[:, 0, 0].tolist() == [1001, 1002]

    def test_bad_zrange(self, fixture_dir):
        with pytest.raises(ValueError, match="ZRange"):
            readAllChannelsVolume(str(fixture_dir), _CH561, ZRange=[0, 2])
        with pytest.raises(ValueError, match="ZRange"):
            readAllChannelsVolume(str(fixture_dir), _CH561, ZRange=[3, 5])

    def test_unknown_channel(self, fixture_dir):
        with pytest.raises(ValueError, match="not found"):
            readAllChannelsVolume(str(fixture_dir), "Ex_488_Em_488F_Ch1")


# ----------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------


class TestReaderAllChannels:
    def test_resolve_root_is_all_channels(self, fixture_dir):
        r = ndr_reader_smartspim()
        info = r.resolveepoch([str(fixture_dir)])
        assert info["source"] == "allChannels"
        assert info["channelName"] == ""
        assert info["tileId"] == ""
        assert info["channelNames"] == [_CH561, _CH640]
        assert info["numSlices"] == 4
        assert info["allChannels"]["voxelSizeUm"] == [4, 4, 4]
        assert r.resolveepoch(str(fixture_dir))["source"] == "allChannels"

    def test_resolve_channel_is_stitched(self, fixture_dir):
        info = ndr_reader_smartspim().resolveepoch([str(fixture_dir), _CH640])
        assert info["source"] == "allChannels"
        assert info["channelNames"] == [_CH640]
        assert info["tile"] is None

    def test_resolve_tile_still_reads_tile(self, fixture_dir):
        info = ndr_reader_smartspim().resolveepoch([str(fixture_dir), _CH561, _TILE1])
        assert info["source"] == "tile"
        assert info["tile"]["height"] == 8
        assert info["allChannels"] is None

    def test_unknown_channel_errors(self, fixture_dir):
        with pytest.raises(ValueError, match="not found"):
            ndr_reader_smartspim().resolveepoch([str(fixture_dir), "Ex_488_Em_488F_Ch1"])

    def test_slice_count_mismatch_errors_for_root(self, fixture_dir):
        (_all_dir(fixture_dir) / "200000_100000_000030_ch1.tiff").unlink()
        r = ndr_reader_smartspim()
        with pytest.raises(ValueError, match="different slice counts"):
            r.resolveepoch([str(fixture_dir)])
        assert r.numframes([str(fixture_dir), _CH640], 1) == 3

    def test_fallback_when_channel_incomplete(self, fixture_dir):
        for p in _all_dir(fixture_dir).glob("*_ch1.tiff"):
            p.unlink()
        r = ndr_reader_smartspim()
        with pytest.raises(ValueError, match="does not pin a channel"):
            r.resolveepoch([str(fixture_dir)])
        with pytest.raises(ValueError, match="does not pin a tile"):
            r.resolveepoch([str(fixture_dir), _CH561])

    def test_framesize_and_type(self, fixture_dir):
        r = ndr_reader_smartspim()
        root = str(fixture_dir)
        assert r.framesize([root], 1) == [106, 13, 2, 1, 4]
        assert r.framesize([root, _CH561], 1) == [106, 13, 1, 1, 4]
        assert r.numframes([root], 1) == 4
        assert r.datatype([root], 1) == "uint16"
        assert r.dimensionorder([root], 1) == "YXCZT"
        assert len(r.frametimes([root], 1)) == 4

    def test_readframes_all_channels(self, fixture_dir):
        frames = ndr_reader_smartspim().readframes([str(fixture_dir)], 1, [1, 2, 4])
        assert frames.shape == (106, 13, 2, 1, 3)
        assert frames.dtype == np.uint16
        assert frames[0, 0, 0, 0, :].tolist() == [1000, 1001, 1003]
        assert frames[8, 3, 1, 0, :].tolist() == [2000, 2001, 2003]

    def test_readframes_unordered_indices(self, fixture_dir):
        frames = ndr_reader_smartspim().readframes([str(fixture_dir), _CH640], 1, [4, 2, 3, 3])
        assert frames.shape == (106, 13, 1, 1, 4)
        assert frames[0, 0, 0, 0, :].tolist() == [2003, 2001, 2002, 2002]

    def test_readframes_select_c(self, fixture_dir):
        frames = ndr_reader_smartspim().readframes([str(fixture_dir)], 1, [2], SelectC=[2])
        assert frames.shape == (106, 13, 1, 1, 1)
        assert frames[0, 0, 0, 0, 0] == 2001

    def test_readframes_default_reads_all(self, fixture_dir):
        frames = ndr_reader_smartspim().readframes([str(fixture_dir)], 1, [])
        assert frames.shape == (106, 13, 2, 1, 4)

    def test_readframes_out_of_range(self, fixture_dir):
        with pytest.raises(IndexError):
            ndr_reader_smartspim().readframes([str(fixture_dir)], 1, [1, 5])

    def test_tile_read_unaffected(self, fixture_dir):
        frames = ndr_reader_smartspim().readframes([str(fixture_dir), _CH640, _TILE1], 1, [1])
        assert frames.shape == (8, 10, 1, 1, 1)
        assert frames[0, 0, 0, 0, 0] == 200
