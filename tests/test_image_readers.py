"""Tests for the image/frame API and the tiffstack and prairieview readers.

Ports of the MATLAB readers ndr.reader.tiffstack and ndr.reader.prairieview,
which build on the frame API added to ndr.reader.base.
"""

import numpy as np
import pytest

from ndr.reader.base import ndr_reader_base
from ndr.reader.prairieview import ndr_reader_prairieview
from ndr.reader.tiffstack import ndr_reader_tiffstack
from ndr.reader_wrapper import ndr_reader

tifffile = pytest.importorskip("tifffile")


@pytest.fixture
def movie(tmp_path):
    """A single multipage TIFF: 5 pages of 8x6 uint16."""
    stack = np.arange(5 * 8 * 6, dtype=np.uint16).reshape(5, 8, 6)
    tifffile.imwrite(tmp_path / "movie.tif", stack)
    return tmp_path, stack


@pytest.fixture
def prairie(tmp_path):
    """A Prairie View epoch: 2 channels x 3 frames plus a legacy .pcf config."""
    for f in range(1, 4):
        for c in (1, 2):
            tifffile.imwrite(
                tmp_path / f"rec_Cycle001_Ch{c}_{f:06d}.tif",
                np.full((5, 4), c * 100 + f, dtype=np.uint16),
            )
    (tmp_path / "rec_Main.pcf").write_text(
        "[Main]\nLines per frame=5\nPixels per line=4\n"
        "Frame period (us)=100000\nTotal images=3\n\n"
        "[Image TimeStamp (us)]\nimg1=0\nimg2=100000\nimg3=200000\n"
    )
    return tmp_path


class TestFrameAPIDefaults:
    """The base class supplies inert defaults for non-image readers."""

    def _concrete(self):
        class _C(ndr_reader_base):
            def readchannels_epochsamples(self, *args, **kwargs):
                raise NotImplementedError

            def readevents_epochsamples_native(self, *args, **kwargs):
                raise NotImplementedError

        return _C()

    def test_defaults(self):
        r = self._concrete()
        assert r.numframes([]) == 0
        assert r.framesize([]) == [0, 0, 0, 0, 0]
        assert r.dimensionorder([]) == "YXCZT"
        assert r.datatype([]) == ""
        assert len(r.frametimes([])) == 0
        assert len(r.readframes([])) == 0

    def test_empty_metadata(self):
        m = ndr_reader_base.emptyimagemetadata()
        assert m["israster"] is False
        assert m["bidirectional"] is False
        for k in ("frame_period", "line_period", "dwell_time"):
            assert np.isnan(m[k])

    def test_selectframeCZ_is_one_based(self):
        frames = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
        assert ndr_reader_base.selectframeCZ(frames, [1, 3], [2]).shape == (2, 3, 2, 1, 6)
        assert ndr_reader_base.selectframeCZ(frames, None, None).shape == frames.shape
        # Channel 1 selects the first channel, not the second.
        np.testing.assert_array_equal(
            ndr_reader_base.selectframeCZ(frames, [1], None)[:, :, 0, :, :],
            frames[:, :, 0, :, :],
        )


class TestTiffstack:
    def test_geometry(self, movie):
        tmp_path, stack = movie
        r = ndr_reader_tiffstack()
        es = [str(tmp_path / "movie.tif")]
        assert r.numframes(es) == 5
        assert r.framesize(es) == [8, 6, 1, 1, 5]
        assert r.dimensionorder(es) == "YXCZT"
        assert r.datatype(es) == "uint16"

    def test_readframes_matches_source(self, movie):
        tmp_path, stack = movie
        r = ndr_reader_tiffstack()
        es = [str(tmp_path / "movie.tif")]
        frames = r.readframes(es, 1, [1, 3, 5])
        assert frames.shape == (8, 6, 1, 1, 3)
        for out_i, src_i in enumerate([0, 2, 4]):
            np.testing.assert_array_equal(frames[:, :, 0, 0, out_i], stack[src_i])

    def test_readframes_defaults_to_all(self, movie):
        tmp_path, _ = movie
        r = ndr_reader_tiffstack()
        assert r.readframes([str(tmp_path / "movie.tif")]).shape[-1] == 5

    def test_clockless_without_sidecar(self, movie):
        tmp_path, _ = movie
        r = ndr_reader_tiffstack()
        es = [str(tmp_path / "movie.tif")]
        assert not r.hasframetimes(es)
        assert [c.type for c in r.epochclock(es)] == ["no_time"]
        assert np.all(np.isnan(r.t0_t1(es)[0]))
        assert np.all(np.isnan(r.frametimes(es, 1, [1, 2])))

    def test_frametimes_sidecar(self, movie):
        tmp_path, _ = movie
        (tmp_path / "movie_frametimes.txt").write_text("\n".join(str(0.1 * i) for i in range(5)))
        r = ndr_reader_tiffstack()
        es = [str(tmp_path / "movie.tif")]
        assert r.hasframetimes(es)
        assert [c.type for c in r.epochclock(es)] == ["dev_local_time"]
        np.testing.assert_allclose(r.frametimes(es, 1, [1, 3, 5]), [0.0, 0.2, 0.4])
        np.testing.assert_allclose(r.t0_t1(es)[0], [0.0, 0.4])

    def test_directory_epoch_orders_files(self, tmp_path):
        for i in range(3):
            tifffile.imwrite(tmp_path / f"f{i}.tif", np.full((4, 4), i, dtype=np.uint8))
        r = ndr_reader_tiffstack()
        es = [str(tmp_path)]
        assert r.numframes(es) == 3
        frames = r.readframes(es)
        assert [int(frames[0, 0, 0, 0, i]) for i in range(3)] == [0, 1, 2]

    def test_anchor_file_resolves_directory(self, tmp_path):
        tifffile.imwrite(tmp_path / "a.tif", np.zeros((2, 2), dtype=np.uint8))
        anchor = tmp_path / "marker.txt"
        anchor.write_text("not an image")
        r = ndr_reader_tiffstack()
        assert r.numframes([str(anchor)]) == 1

    def test_no_tiff_raises(self, tmp_path):
        (tmp_path / "nothing.txt").write_text("x")
        with pytest.raises(ValueError, match="No .tif"):
            ndr_reader_tiffstack().numframes([str(tmp_path / "nothing.txt")])

    def test_channel_api_not_applicable(self, movie):
        tmp_path, _ = movie
        with pytest.raises(NotImplementedError, match="image reader"):
            ndr_reader_tiffstack().readchannels_epochsamples(
                "analog_in", 1, [str(tmp_path / "movie.tif")], 1, 1, 2
            )

    def test_getchannelsepoch(self, movie):
        tmp_path, _ = movie
        channels = ndr_reader_tiffstack().getchannelsepoch([str(tmp_path / "movie.tif")])
        assert channels == [{"name": "image1", "type": "image", "time_channel": None}]

    @pytest.mark.parametrize(
        ("np_dtype", "matlab_class"),
        [
            (np.uint8, "uint8"),
            (np.uint16, "uint16"),
            (np.int16, "int16"),
            (np.float32, "single"),
            (np.float64, "double"),
        ],
    )
    def test_datatype_uses_matlab_class_names(self, tmp_path, np_dtype, matlab_class):
        """datatype() reports the MATLAB class name, for exact string parity."""
        tifffile.imwrite(tmp_path / "d.tif", np.zeros((3, 3), dtype=np_dtype))
        r = ndr_reader_tiffstack()
        assert r.datatype([str(tmp_path / "d.tif")]) == matlab_class
        # readframes still allocates the right numpy dtype from that name.
        assert r.readframes([str(tmp_path / "d.tif")]).dtype == np.dtype(np_dtype)

    def test_numpy_dtype_round_trips(self):
        for name in ("uint8", "uint16", "int16", "single", "double"):
            assert ndr_reader_tiffstack.numpy_dtype(name).itemsize > 0
        assert ndr_reader_tiffstack.numpy_dtype("single") == np.dtype(np.float32)
        assert ndr_reader_tiffstack.numpy_dtype("double") == np.dtype(np.float64)


class TestPrairieview:
    def test_channel_grouping(self, prairie):
        r = ndr_reader_prairieview()
        es = [str(prairie)]
        assert r.framesize(es) == [5, 4, 2, 1, 3]
        assert r.numframes(es) == 3
        assert r.datatype(es) == "uint16"

    def test_readframes_places_channels_on_c_axis(self, prairie):
        r = ndr_reader_prairieview()
        frames = r.readframes([str(prairie)])
        assert frames.shape == (5, 4, 2, 1, 3)
        assert [int(frames[0, 0, 0, 0, i]) for i in range(3)] == [101, 102, 103]
        assert [int(frames[0, 0, 1, 0, i]) for i in range(3)] == [201, 202, 203]

    def test_selectc_reads_only_that_channel(self, prairie):
        r = ndr_reader_prairieview()
        frames = r.readframes([str(prairie)], 1, [2], SelectC=[2])
        assert frames.shape == (5, 4, 1, 1, 1)
        assert int(frames[0, 0, 0, 0, 0]) == 202

    def test_per_frame_timestamps_from_pcf(self, prairie):
        r = ndr_reader_prairieview()
        es = [str(prairie)]
        assert r.hasconfigtimes(es)
        np.testing.assert_allclose(r.frametimes(es), [0.0, 0.1, 0.2])
        np.testing.assert_allclose(r.t0_t1(es)[0], [0.0, 0.2])
        assert [c.type for c in r.epochclock(es)] == ["dev_local_time"]

    def test_raster_metadata(self, prairie):
        m = ndr_reader_prairieview().metadata([str(prairie)])
        assert m["israster"] is True
        assert m["frame_period"] == pytest.approx(0.1)
        assert m["lines_per_frame"] == 5
        assert m["pixels_per_line"] == 4
        # line_period derived from frame_period / lines_per_frame
        assert m["line_period"] == pytest.approx(0.02)

    def test_ragged_channels_raise(self, tmp_path):
        tifffile.imwrite(tmp_path / "r_Cycle001_Ch1_000001.tif", np.zeros((2, 2), np.uint8))
        tifffile.imwrite(tmp_path / "r_Cycle001_Ch1_000002.tif", np.zeros((2, 2), np.uint8))
        tifffile.imwrite(tmp_path / "r_Cycle001_Ch2_000001.tif", np.zeros((2, 2), np.uint8))
        with pytest.raises(ValueError, match="same set of frames"):
            ndr_reader_prairieview().framelayout([str(tmp_path)])

    def test_modern_pvscan_xml(self, tmp_path):
        for f in range(1, 3):
            for c in (1, 2):
                tifffile.imwrite(
                    tmp_path / f"s_Cycle001_Ch{c}_{f:06d}.tif",
                    np.full((4, 4), c * 10 + f, dtype=np.int16),
                )
        (tmp_path / "scan.xml").write_text(
            '<?xml version="1.0"?>\n<PVScan version="5.4.64.700">\n'
            " <PVStateShard>\n"
            '  <PVStateValue key="linesPerFrame" value="4" />\n'
            '  <PVStateValue key="pixelsPerLine" value="4" />\n'
            '  <PVStateValue key="framePeriod" value="0.0625" />\n'
            '  <PVStateValue key="scanLinePeriod" value="0.0150" />\n'
            '  <PVStateValue key="bidirectionalScan" value="True" />\n'
            " </PVStateShard>\n"
            " <Sequence>\n"
            '  <Frame absoluteTime="1.5" index="1" />\n'
            '  <Frame absoluteTime="1.5625" index="2" />\n'
            " </Sequence>\n"
            "</PVScan>"
        )
        r = ndr_reader_prairieview()
        es = [str(tmp_path)]
        assert r.datatype(es) == "int16"
        np.testing.assert_allclose(r.frametimes(es), [1.5, 1.5625])
        m = r.metadata(es)
        assert m["bidirectional"] is True
        # scanLinePeriod is used directly rather than derived
        assert m["line_period"] == pytest.approx(0.015)

    def test_datatype_uses_matlab_class_names(self, tmp_path):
        """prairieview reports the same MATLAB class names as tiffstack."""
        for c in (1, 2):
            tifffile.imwrite(
                tmp_path / f"f_Cycle001_Ch{c}_000001.tif",
                np.zeros((3, 3), dtype=np.float32),
            )
        r = ndr_reader_prairieview()
        assert r.datatype([str(tmp_path)]) == "single"
        assert r.readframes([str(tmp_path)]).dtype == np.dtype(np.float32)

    def test_legacy_mm_xml(self, tmp_path):
        for f in range(1, 3):
            tifffile.imwrite(tmp_path / f"m_Ch1_{f:06d}.tif", np.zeros((3, 3), np.uint8))
        (tmp_path / "cfg.xml").write_text(
            '<?xml version="1.0"?>\n<DataSet>\n'
            " <Lines_Per_Frame>3</Lines_Per_Frame>\n"
            " <Pixels_Per_Line>3</Pixels_Per_Line>\n"
            " <Framerate>16</Framerate>\n"
            " <Dataset_x0020_1><Time>0</Time></Dataset_x0020_1>\n"
            " <Dataset_x0020_2><Time>62.5</Time></Dataset_x0020_2>\n"
            "</DataSet>"
        )
        r = ndr_reader_prairieview()
        es = [str(tmp_path)]
        # Legacy times are milliseconds; frame period comes from 1/Framerate.
        np.testing.assert_allclose(r.frametimes(es), [0.0, 0.0625])
        assert r.metadata(es)["frame_period"] == pytest.approx(0.0625)


class TestWrapperFrameDelegation:
    def test_wrapper_delegates(self, movie):
        tmp_path, stack = movie
        r = ndr_reader("tiffstack")
        es = [str(tmp_path / "movie.tif")]
        assert r.numframes(es) == 5
        assert r.framesize(es) == [8, 6, 1, 1, 5]
        assert r.dimensionorder(es) == "YXCZT"
        assert r.datatype(es) == "uint16"
        assert r.readframes(es).shape == (8, 6, 1, 1, 5)
        assert r.metadata(es)["israster"] is False

    def test_wrapper_resolves_new_reader_types(self):
        assert isinstance(ndr_reader("tif").ndr_reader_base, ndr_reader_tiffstack)
        assert isinstance(ndr_reader("pv").ndr_reader_base, ndr_reader_prairieview)
