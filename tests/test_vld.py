"""Tests for the VH Lab LabView (.vld/.vlh) reader.

Port of the MATLAB ndr.reader.vld tests. Fixtures synthesize both binary
layouts the format allows, so the reader is checked against known data rather
than a recorded artifact.
"""

import numpy as np
import pytest

from ndr.format.vld import readvhlvdatafile, readvhlvheaderfile
from ndr.reader.vld import ndr_reader_vld
from ndr.reader_wrapper import ndr_reader

SR = 100.0
NCH = 3
SPC = 10
NCHUNKS = 4
TOTAL = SPC * NCHUNKS


def _truth():
    return np.arange(TOTAL * NCH, dtype=np.float64).reshape(TOTAL, NCH)


def _write_header(path, multiplexed, extra=""):
    path.write_text(
        "ChannelString:\t/dev/ai0,/dev/ai1,/dev/ai2\n"
        f"NumChans:\t{NCH}\n"
        f"SamplingRate:\t{int(SR)}\n"
        f"SamplesPerChunk:\t{SPC}\n"
        f"Multiplexed:\t{int(multiplexed)}\n" + extra
    )


@pytest.fixture
def chunked(tmp_path):
    """Chunked layout: SamplesPerChunk of ch1, then ch2, ... per chunk."""
    truth = _truth()
    with open(tmp_path / "rec.vld", "wb") as f:
        for c in range(NCHUNKS):
            blk = truth[c * SPC : (c + 1) * SPC, :]
            for ch in range(NCH):
                f.write(blk[:, ch].astype(">f8").tobytes())
    _write_header(tmp_path / "rec.vlh", multiplexed=False)
    return tmp_path / "rec.vld", truth


@pytest.fixture
def muxed(tmp_path):
    """Multiplexed layout: samples interleaved across channels."""
    truth = _truth()
    (tmp_path / "rec.vld").write_bytes(truth.astype(">f8").tobytes())
    _write_header(tmp_path / "rec.vlh", multiplexed=True)
    return tmp_path / "rec.vld", truth


class TestHeader:
    def test_parses_fields(self, chunked):
        vld, _ = chunked
        h = readvhlvheaderfile(vld.with_suffix(".vlh"))
        assert h["NumChans"] == NCH
        assert h["SamplingRate"] == int(SR)
        assert h["SamplesPerChunk"] == SPC
        assert h["Multiplexed"] == 0
        assert h["ChannelString"] == "/dev/ai0,/dev/ai1,/dev/ai2"

    def test_rejects_data_file(self, chunked):
        vld, _ = chunked
        with pytest.raises(ValueError, match="data file"):
            readvhlvheaderfile(vld)


@pytest.mark.parametrize("layout", ["chunked", "muxed"])
class TestReadBothLayouts:
    def _fixture(self, layout, request):
        return request.getfixturevalue(layout)

    def test_full_read_matches(self, layout, request):
        vld, truth = self._fixture(layout, request)
        r = ndr_reader_vld()
        got = r.readchannels_epochsamples("analog_in", [1, 2, 3], [str(vld)], 1, 1, TOTAL)
        assert got.shape == (TOTAL, NCH)
        np.testing.assert_allclose(got, truth)

    def test_partial_channel_read(self, layout, request):
        vld, truth = self._fixture(layout, request)
        r = ndr_reader_vld()
        got = r.readchannels_epochsamples("analog_in", [2], [str(vld)], 1, 5, 14)
        np.testing.assert_allclose(got.ravel(), truth[4:14, 1])

    def test_time_channel(self, layout, request):
        vld, _ = self._fixture(layout, request)
        r = ndr_reader_vld()
        t = r.readchannels_epochsamples("time", [1], [str(vld)], 1, 1, TOTAL)
        np.testing.assert_allclose(t.ravel(), np.arange(TOTAL) / SR)

    def test_t0_t1(self, layout, request):
        vld, _ = self._fixture(layout, request)
        np.testing.assert_allclose(ndr_reader_vld().t0_t1([str(vld)])[0], [0.0, (TOTAL - 1) / SR])

    def test_format_function_directly(self, layout, request):
        vld, truth = self._fixture(layout, request)
        h = readvhlvheaderfile(vld.with_suffix(".vlh"))
        T, D, tot_sam, tot_time = readvhlvdatafile(vld, h, [1, 2, 3], 0.0, (TOTAL - 1) / SR)
        assert tot_sam == TOTAL
        assert tot_time == pytest.approx(TOTAL / SR)
        np.testing.assert_allclose(D, truth)
        np.testing.assert_allclose(T, np.arange(TOTAL) / SR)


class TestReaderInterface:
    def test_channels_named_positionally(self, chunked):
        vld, _ = chunked
        channels = ndr_reader_vld().getchannelsepoch([str(vld)])
        assert [c["name"] for c in channels] == ["t1", "ai1", "ai2", "ai3"]
        assert channels[0]["type"] == "time"
        assert all(c["type"] == "analog_in" for c in channels[1:])

    def test_samplerate(self, chunked):
        vld, _ = chunked
        assert ndr_reader_vld().samplerate([str(vld)], 1, "ai", 1) == SR

    def test_epochclock(self, chunked):
        vld, _ = chunked
        assert [c.type for c in ndr_reader_vld().epochclock([str(vld)])] == ["dev_local_time"]

    def test_labeling_convention_is_indexed(self):
        assert ndr_reader_vld().channelLabelingConvention("analog_in") == "indexed"

    def test_uniform_list_channeltype(self, chunked):
        vld, truth = chunked
        r = ndr_reader_vld()
        scalar = r.readchannels_epochsamples("analog_in", [1, 2], [str(vld)], 1, 1, 10)
        as_list = r.readchannels_epochsamples(
            ["analog_in", "analog_in"], [1, 2], [str(vld)], 1, 1, 10
        )
        np.testing.assert_array_equal(scalar, as_list)

    def test_heterogeneous_list_raises(self, chunked):
        vld, _ = chunked
        with pytest.raises(ValueError, match="uniform"):
            ndr_reader_vld().readchannels_epochsamples(
                ["analog_in", "time"], [1, 2], [str(vld)], 1, 1, 10
            )

    def test_missing_vld_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No file ending with ".vld"'):
            ndr_reader_vld().filenamefromepochfiles([str(tmp_path / "x.txt")])

    def test_epoch_select_beyond_available_raises(self, chunked):
        vld, _ = chunked
        with pytest.raises(ValueError, match="epoch_select cannot be 2"):
            ndr_reader_vld().filenamefromepochfiles([str(vld)], 2)

    def test_out_of_range_channel_raises(self, chunked):
        vld, _ = chunked
        h = readvhlvheaderfile(vld.with_suffix(".vlh"))
        with pytest.raises(ValueError, match="between 1 and NumChans"):
            readvhlvdatafile(vld, h, [99], 0.0, 0.1)

    def test_daqchannels2internalchannels(self, chunked):
        vld, _ = chunked
        cs = ndr_reader_vld().daqchannels2internalchannels(["ai", "ai"], [1, 3], [str(vld)])
        assert [c["internal_channelname"] for c in cs] == ["ai1", "ai3"]
        assert all(c["internal_type"] == "analog_in" for c in cs)
        assert all(c["samplerate"] == SR for c in cs)

    def test_wrapper_resolves_vld(self):
        assert isinstance(ndr_reader("vld").ndr_reader_base, ndr_reader_vld)


class TestScaledIntegers:
    def test_int16_with_scale(self, tmp_path):
        raw = np.arange(-10, 10, dtype=np.int16).reshape(10, 2)
        (tmp_path / "s.vld").write_bytes(raw.astype(">i2").tobytes())
        (tmp_path / "s.vlh").write_text(
            "NumChans:\t2\nSamplingRate:\t100\nSamplesPerChunk:\t5\n"
            "Multiplexed:\t1\nprecision:\tint16\nScale:\t10\n"
        )
        r = ndr_reader_vld()
        got = r.readchannels_epochsamples("analog_in", [1, 2], [str(tmp_path / "s.vld")], 1, 1, 10)
        expect = raw.astype(np.float32) * (10.0 / (2**15 - 1))
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, expect, atol=1e-9)

    def test_underlying_datatype_reports_scale(self, tmp_path):
        (tmp_path / "s.vld").write_bytes(np.zeros((10, 2), dtype=">i2").tobytes())
        (tmp_path / "s.vlh").write_text(
            "NumChans:\t2\nSamplingRate:\t100\nSamplesPerChunk:\t5\n"
            "Multiplexed:\t1\nprecision:\tint16\nScale:\t10\n"
        )
        dt, p, size = ndr_reader_vld().underlying_datatype([str(tmp_path / "s.vld")], 1, "ai", 1)
        assert dt == "int16"
        assert size == 16
        assert p[0][1] == pytest.approx(10.0 / (2**15 - 1))
