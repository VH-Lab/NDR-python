"""Tests for the native video reader.

Port of NDR-matlab tools/tests/+ndr/+unittest/+reader/TestVideo.m. Small
synthetic videos with known per-frame content are written with PyAV into a
temporary folder:

  - an RGB video, written as MPEG-4 and as Motion JPEG AVI (the two profiles
    the MATLAB test tries). Both are lossy, so pixel content is compared
    through per-frame, per-channel means with a tolerance;
  - a greyscale video written as uncompressed 8-bit palettised AVI with a grey
    palette -- what MATLAB's 'Grayscale AVI' profile writes -- which checks the
    C=1 path and exact pixel round-trip.

Beyond the MATLAB test, a variable-frame-rate file checks that frame times are
measured rather than assumed, and a Matroska file (whose header carries no
frame count) checks the counting fallback and 16-bit greyscale.
"""

from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from ndr.reader.video import ndr_reader_video  # noqa: E402
from ndr.reader_wrapper import ndr_reader  # noqa: E402

Y, X, T = 24, 32, 12
FRAMERATE = 4
MEANTOL = 6  # tolerance on per-frame channel means for lossy codecs


def _write(filename, codec, pix_fmt, frames, fmt_in, rate=FRAMERATE, times=None):
    """Write Y x X (x C) frames; ``times`` (seconds) gives explicit timestamps."""
    with av.open(str(filename), "w") as c:
        s = c.add_stream(codec, rate=rate)
        s.width, s.height, s.pix_fmt = X, Y, pix_fmt
        if times is not None:
            s.codec_context.time_base = Fraction(1, 1000)
        for i, f in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format=fmt_in)
            if times is not None:
                vf.pts = round(times[i] * 1000)
                vf.time_base = Fraction(1, 1000)
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return str(filename)


def _rgbtruth(n=T):
    """Flat frames with a frame-dependent level; the three channels differ."""
    rgb = np.zeros((Y, X, 3, 1, n), dtype=np.uint8)
    for i in range(n):
        level = 20 + i * 18  # 20 .. 218
        rgb[:, :, 0, 0, i] = level
        rgb[:, :, 1, 0, i] = 255 - level
        rgb[:, :, 2, 0, i] = 128
    return rgb


def _greytruth():
    """A spatial gradient plus a frame-dependent offset."""
    base = np.arange(Y * X).reshape(X, Y).T  # MATLAB column-major reshape
    grey = np.zeros((Y, X, 1, 1, T), dtype=np.uint8)
    for i in range(T):
        grey[:, :, 0, 0, i] = np.mod(base + i * 7, 200)
    return grey


def _writegreyavi(filename, grey):
    """Raw 8-bit palettised AVI with a grey palette, as MATLAB 'Grayscale AVI'."""
    with av.open(str(filename), "w") as c:
        s = c.add_stream("rawvideo", rate=FRAMERATE)
        s.width, s.height, s.pix_fmt = X, Y, "pal8"
        # FFmpeg palette entries are uint32 ARGB (bytes B,G,R,A); index k is grey k
        palette = np.repeat(np.arange(256, dtype=np.uint8), 4).reshape(256, 4)
        palette[:, 3] = 255
        for i in range(grey.shape[4]):
            vf = av.VideoFrame(X, Y, "pal8")
            idxplane = vf.planes[0]
            rows = np.zeros((Y, idxplane.line_size), dtype=np.uint8)
            rows[:, :X] = grey[:, :, 0, 0, i]
            idxplane.update(rows.tobytes())
            vf.planes[1].update(palette.tobytes())
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return str(filename)


def _channelmeans(frames):
    """C x T matrix of per-frame, per-channel means of a YXCZT array."""
    return frames.astype(float).mean(axis=(0, 1))[:, 0, :]


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("ndr_video_test")
    rgb = _rgbtruth()
    frames = [rgb[:, :, :, 0, i] for i in range(T)]
    grey = _greytruth()
    return {
        "dir": d,
        "rgb": rgb,
        "grey": grey,
        "mp4": _write(d / "rgbvideo.mp4", "mpeg4", "yuv420p", frames, "rgb24"),
        "avi": _write(d / "rgbvideo.avi", "mjpeg", "yuvj420p", frames, "rgb24"),
        "greyfile": _writegreyavi(d / "greyvideo.avi", grey),
    }


@pytest.fixture(params=["mp4", "avi"])
def rgbfile(request, videos):
    return videos[request.param]


@pytest.fixture
def reader():
    return ndr_reader("video")


class TestVideoReader:
    def test_known_readers(self, reader):
        assert isinstance(reader.ndr_reader_base, ndr_reader_video)
        for alias in ("mp4", "avi", "mov"):
            assert isinstance(ndr_reader(alias).ndr_reader_base, ndr_reader_video)

    def test_greyscale_avi_is_palettised_like_matlab(self, videos):
        # the fixture really is MATLAB's 'Grayscale AVI' storage, not plain gray
        with av.open(videos["greyfile"]) as c:
            assert c.streams.video[0].codec_context.pix_fmt == "pal8"

    def test_numframes_and_size(self, reader, rgbfile):
        ef = [rgbfile]
        assert reader.numframes(ef, 1) == T
        assert reader.framesize(ef, 1) == [Y, X, 3, 1, T]

    def test_dimensionorder_and_datatype(self, reader, rgbfile):
        ef = [rgbfile]
        assert reader.dimensionorder(ef, 1) == "YXCZT"
        assert reader.datatype(ef, 1) == "uint8"

    def test_clock_and_times(self, reader, rgbfile):
        ef = [rgbfile]
        ec = reader.epochclock(ef, 1)
        assert len(ec) == 1
        assert ec[0].type == "dev_local_time"

        ft = reader.frametimes(ef, 1)
        assert ft.shape == (T,)
        np.testing.assert_allclose(ft, np.arange(T) / FRAMERATE, atol=1e-3)
        np.testing.assert_allclose(np.diff(ft), 1 / FRAMERATE, atol=1e-3)

        np.testing.assert_array_equal(reader.frametimes(ef, 1, [2, 5, 9]), ft[[1, 4, 8]])

        t0t1 = reader.t0_t1(ef, 1)
        np.testing.assert_allclose(t0t1[0], [ft[0], ft[-1]], atol=1e-9)

    def test_t0_t1_without_a_full_scan(self, reader, videos, rgbfile):
        # a fresh copy, so neither the scan's cache nor endtimes' has seen it;
        # it must agree with the full scan, taken after
        for src in (rgbfile, videos["greyfile"]):
            src = videos["dir"] / src.split("/")[-1]
            copy = videos["dir"] / f"fresh_{src.stem}{src.suffix}"
            copy.write_bytes(src.read_bytes())
            t0t1 = reader.t0_t1([str(copy)], 1)
            np.testing.assert_allclose(t0t1[0], [0, (T - 1) / FRAMERATE], atol=1e-3)
            ft = reader.frametimes([str(copy)], 1)
            np.testing.assert_allclose(t0t1[0], [ft[0], ft[-1]], atol=1e-9)
            copy.unlink()

    @pytest.mark.parametrize(
        "codec,pix_fmt,ext", [("mpeg4", "yuv420p", ".mp4"), ("mjpeg", "yuvj420p", ".avi")]
    )
    def test_endtimes_of_a_longer_video(self, videos, codec, pix_fmt, ext):
        # longer than the seek margin (5 frames): the last frame is reached
        # by seeking, not by decoding from the start
        n = 60
        rgb = np.repeat(_rgbtruth(1), n, axis=4)
        for i in range(n):
            rgb[:, :, 0, 0, i] = np.mod((i + 1) * 4, 250)
        name = f"longvideo{ext}"
        _write(
            videos["dir"] / name, codec, pix_fmt, [rgb[:, :, :, 0, i] for i in range(n)], "rgb24"
        )
        folder = str(videos["dir"])
        t0t1 = ndr_reader_video.endtimes(folder, name)
        ft = ndr_reader_video.scanframetimes(folder, name)
        assert len(ft) == n
        np.testing.assert_allclose(t0t1, [ft[0], ft[-1]], atol=1e-9)

    def test_frames_content(self, reader, videos, rgbfile):
        frames = reader.readframes([rgbfile], 1)
        assert frames.shape == (Y, X, 3, 1, T)
        assert frames.dtype == np.uint8
        np.testing.assert_allclose(
            _channelmeans(frames), _channelmeans(videos["rgb"]), atol=MEANTOL
        )

    def test_random_access_subset(self, reader, videos, rgbfile):
        ef = [rgbfile]
        all_frames = reader.readframes(ef, 1)
        idx = [7, 2, 3, 4, 12, 1]  # out of order, with a consecutive run
        subset = reader.readframes(ef, 1, idx)
        assert subset.shape == (Y, X, 3, 1, len(idx))
        sel = np.asarray(idx) - 1
        np.testing.assert_array_equal(subset, all_frames[:, :, :, :, sel])
        np.testing.assert_allclose(
            _channelmeans(subset), _channelmeans(videos["rgb"][:, :, :, :, sel]), atol=MEANTOL
        )
        one = reader.readframes(ef, 1, [5])
        np.testing.assert_array_equal(one, all_frames[:, :, :, :, [4]])

    def test_selectc(self, reader, rgbfile):
        ef = [rgbfile]
        g = reader.readframes(ef, 1, [1, 2], SelectC=[2])
        assert g.shape == (Y, X, 1, 1, 2)
        np.testing.assert_array_equal(g, reader.readframes(ef, 1, [1, 2])[:, :, [1], :, :])

    def test_greyscale_video(self, reader, videos):
        ef = [videos["greyfile"]]
        grey = videos["grey"]
        assert reader.numframes(ef, 1) == T
        assert reader.framesize(ef, 1) == [Y, X, 1, 1, T]
        assert reader.datatype(ef, 1) == "uint8"
        np.testing.assert_array_equal(reader.readframes(ef, 1), grey)
        np.testing.assert_array_equal(reader.readframes(ef, 1, [10, 3]), grey[:, :, :, :, [9, 2]])
        np.testing.assert_allclose(reader.frametimes(ef, 1), np.arange(T) / FRAMERATE, atol=1e-3)

    def test_companion_files_ignored(self, reader, videos):
        companion = videos["dir"] / "epochprobemap.ndi"
        companion.write_text("name reference type\n")
        ef = [str(companion), videos["greyfile"]]
        assert reader.numframes(ef, 1) == T
        np.testing.assert_array_equal(
            reader.readframes(ef, 1, [1]), videos["grey"][:, :, :, :, [0]]
        )

    def test_directory_epoch(self, tmp_path, videos):
        (tmp_path / "clip.avi").write_bytes(open(videos["greyfile"], "rb").read())
        (tmp_path / "epochprobemap.ndi").write_text("name reference type\n")
        assert ndr_reader_video().numframes([str(tmp_path)]) == T

    def test_exactly_one_video_file(self, reader, videos):
        with pytest.raises(ValueError, match="exactly one video file"):
            reader.numframes([videos["mp4"], videos["greyfile"]], 1)
        txt = videos["dir"] / "notes.txt"
        txt.write_text("no video here\n")
        with pytest.raises(ValueError, match="No video file found"):
            reader.numframes([str(txt)], 1)

    def test_bad_frame_index(self, reader, videos):
        ef = [videos["greyfile"]]
        with pytest.raises(IndexError):
            reader.readframes(ef, 1, [T + 1])
        with pytest.raises(IndexError):
            reader.frametimes(ef, 1, [0])
        with pytest.raises(IndexError):
            reader.frametimes(ef, 1, [1.5])

    def test_channels_and_metadata(self, reader, rgbfile):
        ef = [rgbfile]
        channels = reader.getchannelsepoch(ef, 1)
        assert len(channels) == 1
        assert channels[0]["type"] == "image"
        assert channels[0]["name"] == "image1"
        m = reader.metadata(ef, 1)
        assert m["israster"] is False
        assert m["frame_period"] == pytest.approx(1 / FRAMERATE, abs=1e-6)

    def test_channel_api_not_applicable(self, videos):
        with pytest.raises(NotImplementedError):
            ndr_reader_video().readchannels_epochsamples("ai", 1, [videos["mp4"]], 1, 1, 10)


class TestVideoReaderBeyondMatlab:
    def test_variable_frame_rate_times_are_measured(self, tmp_path):
        # (k-1)/FrameRate would be wrong here: the gaps are uneven
        times = [0.0, 0.1, 0.15, 0.5, 0.55, 1.2]
        rgb = _rgbtruth(len(times))
        f = _write(
            tmp_path / "vfr.mkv",
            "mjpeg",
            "yuvj420p",
            [rgb[:, :, :, 0, i] for i in range(len(times))],
            "rgb24",
            times=times,
        )
        r = ndr_reader_video()
        np.testing.assert_allclose(r.frametimes([f]), times, atol=1e-3)
        np.testing.assert_allclose(r.t0_t1([f])[0], [0.0, 1.2], atol=1e-3)
        # random access lands on the right frames despite the uneven gaps
        sub = r.readframes([f], 1, [5, 2])
        np.testing.assert_allclose(
            _channelmeans(sub), _channelmeans(rgb[:, :, :, :, [4, 1]]), atol=MEANTOL
        )

    def test_no_header_frame_count_and_16bit_grey(self, tmp_path, videos):
        # Matroska carries no frame count, so numframes counts by decoding
        grey16 = videos["grey"].astype(np.uint16) * 300
        f = _write(
            tmp_path / "grey16.mkv",
            "ffv1",
            "gray16le",
            [grey16[:, :, 0, 0, i] for i in range(T)],
            "gray16le",
        )
        r = ndr_reader_video()
        with r.videoobject([f]) as v:
            assert np.isnan(ndr_reader_video.videonumframes(v))
        assert r.numframes([f]) == T
        assert r.datatype([f]) == "uint16"
        assert r.framesize([f]) == [Y, X, 1, 1, T]
        np.testing.assert_array_equal(r.readframes([f]), grey16)
        np.testing.assert_array_equal(r.readframes([f], 1, [11, 4, 5]), grey16[..., [10, 3, 4]])

    def test_scan_is_cached_and_invalidated_by_a_change(self, tmp_path, videos):
        f = tmp_path / "clip.avi"
        f.write_bytes(open(videos["greyfile"], "rb").read())
        t1 = ndr_reader_video.scanframetimes(str(tmp_path), "clip.avi")
        assert len(t1) == T
        rgb = _rgbtruth(5)
        _write(f, "mjpeg", "yuvj420p", [rgb[:, :, :, 0, i] for i in range(5)], "rgb24")
        assert len(ndr_reader_video.scanframetimes(str(tmp_path), "clip.avi")) == 5

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ndr_reader_video.scanframetimes(str(tmp_path), "absent.mp4")
