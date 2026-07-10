"""Unit tests for the ndr.reader.tiffstack image/frame API.

Ported from tools/tests/+ndr/+unittest/+reader/TestTiffstack.m

Verifies the frame-based reading interface of ndr.reader.tiffstack
(numframes, framesize, dimensionorder, datatype, frametimes, readframes,
epochclock, t0_t1, getchannelsepoch), independently of NDI. Programmatically
generates temporary multipage TIFF stacks with known pixel contents and
checks the reader against them.

Cases (mirroring TestTiffstack.m):
  (a) a multipage clockless stack   -> numframes/framesize/datatype + exact
      round-trip of the written frames (and a subset);
  (b) a movie stack with a '<name>_frametimes.txt' sidecar -> dev_local_time,
      t0_t1 [first last], frametimes match the sidecar;
  (c) a directory of single-frame TIFFs (Prairie-like), with a 'frametimes.txt'
      sidecar and a non-image '.xml' companion that must be ignored ->
      discovered + read in name order; directory == explicit-file-list ==
      anchor-file resolution.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from ndr.reader.tiffstack import ndr_reader_tiffstack

Y = 8  # image height
X = 6  # image width
T = 5  # number of frames (pages)


def _truth() -> np.ndarray:
    """Y x X x 1 x 1 x T uint16 ground-truth stack (matches TestTiffstack.m).

    truth(:,:,1,1,i) = uint16( reshape(1:(Y*X), Y, X) + (i-1)*1000 )

    MATLAB reshape is column-major; numpy default is row-major, so we build
    the per-frame base with order='F' to match the MATLAB pixel layout.
    """
    truth = np.zeros((Y, X, 1, 1, T), dtype=np.uint16)
    base = np.arange(1, Y * X + 1).reshape(Y, X, order="F")
    for i in range(T):
        truth[:, :, 0, 0, i] = (base + i * 1000).astype(np.uint16)
    return truth


def _write_multipage_tiff(filename: str, data: np.ndarray) -> None:
    """Write a Y x X x C x 1 x T array as a multipage TIFF (page per frame)."""
    n = data.shape[4]
    pages = [data[:, :, 0, 0, i] for i in range(n)]
    # tifffile writes each 2D array in the list as its own page/directory.
    tifffile.imwrite(filename, np.stack(pages, axis=0))


@pytest.fixture()
def stack(tmp_path):
    """Build all fixture stacks and return a namespace-like dict."""
    truth = _truth()

    clockless = tmp_path / "stack_clockless.tif"
    _write_multipage_tiff(str(clockless), truth)

    movie = tmp_path / "stack_movie.tif"
    _write_multipage_tiff(str(movie), truth)
    movie_times = np.arange(T) * 0.1 + 10.0
    np.savetxt(str(tmp_path / "stack_movie_frametimes.txt"), movie_times, fmt="%.10g")

    dir_epoch = tmp_path / "prairie_like"
    dir_epoch.mkdir()
    dir_files = []
    for i in range(T):
        fn = dir_epoch / f"frame_{i + 1:05d}.tif"
        _write_multipage_tiff(str(fn), truth[:, :, :, :, i : i + 1])
        dir_files.append(str(fn))
    dir_times = np.arange(T) * 0.25 + 5.0
    np.savetxt(str(dir_epoch / "frametimes.txt"), dir_times, fmt="%.10g")
    # a non-image companion that must be ignored / used only as an anchor
    (dir_epoch / "metadata.xml").write_text("1\n2\n3\n")

    return {
        "reader": ndr_reader_tiffstack(),
        "truth": truth,
        "clockless": str(clockless),
        "movie": str(movie),
        "movie_times": movie_times,
        "dir_epoch": str(dir_epoch),
        "dir_files": dir_files,
        "dir_times": dir_times,
    }


# --------------------------------------------------------------------------
# (a) multipage clockless stack
# --------------------------------------------------------------------------


def test_numframes_and_size(stack):
    r = stack["reader"]
    ef = [stack["clockless"]]
    assert r.numframes(ef, 1) == T
    assert r.framesize(ef, 1) == [Y, X, 1, 1, T]


def test_dimensionorder_and_datatype(stack):
    r = stack["reader"]
    ef = [stack["clockless"]]
    assert r.dimensionorder(ef, 1) == "YXCZT"
    assert r.datatype(ef, 1) == "uint16"


def test_clockless_clock_and_times(stack):
    r = stack["reader"]
    ef = [stack["clockless"]]
    ec = r.epochclock(ef, 1)
    assert len(ec) == 1
    assert ec[0].type == "no_time"
    t0t1 = r.t0_t1(ef, 1)
    assert all(np.isnan(t0t1[0]))
    ft = r.frametimes(ef, 1)
    assert np.all(np.isnan(ft)) and ft.size == T


def test_clockless_frames_round_trip(stack):
    r = stack["reader"]
    ef = [stack["clockless"]]
    frames = r.readframes(ef, 1)
    assert frames.dtype == np.uint16
    assert np.array_equal(frames, stack["truth"])
    # subset (1-based indices, mirroring MATLAB [2 4])
    subset = r.readframes(ef, 1, [2, 4])
    assert np.array_equal(subset, stack["truth"][:, :, :, :, [1, 3]])


def test_get_channels_epoch(stack):
    r = stack["reader"]
    channels = r.getchannelsepoch([stack["clockless"]], 1)
    assert len(channels) == 1
    assert channels[0]["type"] == "image"


# --------------------------------------------------------------------------
# (b) movie stack (frame-times sidecar)
# --------------------------------------------------------------------------


def test_movie_clock_and_times(stack):
    r = stack["reader"]
    ef = [stack["movie"]]
    ec = r.epochclock(ef, 1)
    assert ec[0].type == "dev_local_time"
    t0t1 = r.t0_t1(ef, 1)
    assert t0t1[0] == [stack["movie_times"][0], stack["movie_times"][-1]]
    ft = r.frametimes(ef, 1)
    assert np.allclose(ft, stack["movie_times"])
    ftsub = r.frametimes(ef, 1, [1, 3, 5])
    assert np.allclose(ftsub, stack["movie_times"][[0, 2, 4]])


def test_movie_frames_round_trip(stack):
    r = stack["reader"]
    frames = r.readframes([stack["movie"]], 1)
    assert np.array_equal(frames, stack["truth"])


# --------------------------------------------------------------------------
# (c) directory of single-frame TIFFs + anchor resolution
# --------------------------------------------------------------------------


def test_directory_epoch_geometry_and_frames(stack):
    r = stack["reader"]
    ef = [stack["dir_epoch"]]
    assert r.numframes(ef, 1) == T
    assert r.framesize(ef, 1) == [Y, X, 1, 1, T]
    frames = r.readframes(ef, 1)
    assert np.array_equal(frames, stack["truth"])
    subset = r.readframes(ef, 1, [1, 3])
    assert np.array_equal(subset, stack["truth"][:, :, :, :, [0, 2]])


def test_directory_epoch_times(stack):
    r = stack["reader"]
    ef = [stack["dir_epoch"]]
    ec = r.epochclock(ef, 1)
    assert ec[0].type == "dev_local_time"
    ft = r.frametimes(ef, 1)
    assert np.allclose(ft, stack["dir_times"])
    t0t1 = r.t0_t1(ef, 1)
    assert t0t1[0] == [stack["dir_times"][0], stack["dir_times"][-1]]


def test_directory_and_file_list_agree(stack):
    r = stack["reader"]
    by_dir = r.readframes([stack["dir_epoch"]], 1)
    by_list = r.readframes(stack["dir_files"], 1)
    assert np.array_equal(by_dir, by_list)


def test_anchor_file_resolves_to_directory(stack):
    r = stack["reader"]
    anchor = str(np.array([stack["dir_epoch"]])[0]) + "/metadata.xml"
    by_anchor = r.readframes([anchor], 1)
    by_dir = r.readframes([stack["dir_epoch"]], 1)
    assert np.array_equal(by_anchor, by_dir)
    ft_anchor = r.frametimes([anchor], 1)
    assert np.allclose(ft_anchor, stack["dir_times"])
    ec = r.epochclock([anchor], 1)
    assert ec[0].type == "dev_local_time"
