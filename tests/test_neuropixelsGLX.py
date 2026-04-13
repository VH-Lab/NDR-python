"""Tests for Neuropixels SpikeGLX reader — AP and NIDQ stream support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ndr.format.neuropixelsGLX.header import header
from ndr.reader.neuropixelsGLX import ndr_reader_neuropixelsGLX

# ---------------------------------------------------------------------------
# Fixtures: create minimal .meta / .bin files on disk
# ---------------------------------------------------------------------------


NIDQ_META_CONTENT = """\
acqMnMaXaDw=0,0,8,1
appVersion=20231207
fileCreateTime=2026-02-10T12:35:42
fileSizeBytes=618204852
fileTimeSecs=3242.1410015948127
firstSample=501549
nSavedChans=9
niAiRangeMax=5
niAiRangeMin=-5
niMaxInt=32768
niSampRate=10593.220339
niXDBytes1=1
niXDChans1=0:7
snsMnMaXaDw=0,0,8,1
snsSaveChanSubset=all
typeThis=nidq
~snsChanMap=(0,0,32,8,1)(XA0;0:0)(XA1;1:1)(XA2;2:2)(XA3;3:3)(XA4;4:4)(XA5;5:5)(XA6;6:6)(XA7;7:7)(XD0;8:8)
"""

AP_META_CONTENT = """\
appVersion=20231207
fileCreateTime=2026-02-10T12:35:42
fileSizeBytes=1000000
fileTimeSecs=1.0
imAiRangeMax=0.6
imAiRangeMin=-0.6
imDatPrb_type=0
imDatPrb_sn=12345
imMaxInt=512
imSampRate=30000
nSavedChans=385
snsApLfSy=384,0,1
snsSaveChanSubset=all
typeThis=imec
"""


def _write_meta(directory: Path, basename: str, content: str) -> Path:
    meta_path = directory / basename
    meta_path.write_text(content)
    return meta_path


def _write_bin(directory: Path, basename: str, n_chans: int, n_samples: int) -> Path:
    """Write a minimal int16 binary file with known data."""
    bin_path = directory / basename
    data = np.zeros((n_samples, n_chans), dtype=np.int16)
    # Fill analog channels with ascending values per channel
    for c in range(n_chans - 1):  # last column(s) = digital word
        data[:, c] = np.arange(1, n_samples + 1, dtype=np.int16) * (c + 1)
    # Digital word column: pack known bit pattern
    # Set bit 0 on odd samples, bit 1 on every-4th sample, etc.
    for s in range(n_samples):
        dw = 0
        if s % 2 == 1:
            dw |= 1  # bit 0
        if s % 4 == 0:
            dw |= 2  # bit 1
        if s % 8 == 0:
            dw |= 4  # bit 2
        data[s, n_chans - 1] = np.int16(dw)
    data.tofile(bin_path)
    return bin_path


@pytest.fixture
def nidq_files(tmp_path):
    """Create NIDQ .meta + .bin in a temp directory (9 channels: 8 XA + 1 DW)."""
    n_chans = 9
    n_samples = 100
    meta = _write_meta(tmp_path, "run_g0_t0.nidq.meta", NIDQ_META_CONTENT)
    binf = _write_bin(tmp_path, "run_g0_t0.nidq.bin", n_chans, n_samples)
    return str(meta), str(binf), n_chans, n_samples


@pytest.fixture
def ap_files(tmp_path):
    """Create AP .meta + .bin in a temp directory (385 channels: 384 AP + 1 SY)."""
    n_chans = 385
    n_samples = 50
    meta = _write_meta(tmp_path, "run_g0_t0.imec0.ap.meta", AP_META_CONTENT)
    binf = _write_bin(tmp_path, "run_g0_t0.imec0.ap.bin", n_chans, n_samples)
    return str(meta), str(binf), n_chans, n_samples


# ---------------------------------------------------------------------------
# header.py tests
# ---------------------------------------------------------------------------


class TestHeaderNIDQ:
    def test_stream_type(self, nidq_files):
        meta_path, _, _, _ = nidq_files
        info = header(meta_path)
        assert info["stream_type"] == "nidq"

    def test_sample_rate(self, nidq_files):
        meta_path, _, _, _ = nidq_files
        info = header(meta_path)
        assert info["sample_rate"] == pytest.approx(10593.220339)

    def test_channel_counts(self, nidq_files):
        meta_path, _, _, _ = nidq_files
        info = header(meta_path)
        assert info["n_saved_chans"] == 9
        assert info["n_mn_chans"] == 0
        assert info["n_ma_chans"] == 0
        assert info["n_xa_chans"] == 8
        assert info["n_dw_chans"] == 1
        assert info["n_neural_chans"] == 8  # MN + MA + XA
        assert info["n_sync_chans"] == 1  # DW count

    def test_digital_lines(self, nidq_files):
        meta_path, _, _, _ = nidq_files
        info = header(meta_path)
        # niXDBytes1=1 → 8 digital lines
        assert info["n_digital_lines"] == 8
        assert info["n_digital_word_cols"] == 1
        assert len(info["digital_line_col"]) == 8
        assert len(info["digital_line_bit"]) == 8
        assert len(info["digital_line_label"]) == 8
        # All 8 lines should be in column 0
        np.testing.assert_array_equal(info["digital_line_col"], 0)
        # Bits should be 0..7
        np.testing.assert_array_equal(info["digital_line_bit"], np.arange(8))
        # Labels
        assert info["digital_line_label"][0] == "XD0"
        assert info["digital_line_label"][7] == "XD7"

    def test_voltage_range(self, nidq_files):
        meta_path, _, _, _ = nidq_files
        info = header(meta_path)
        assert info["voltage_range"] == (-5.0, 5.0)

    def test_max_int(self, nidq_files):
        meta_path, _, _, _ = nidq_files
        info = header(meta_path)
        assert info["max_int"] == 32768

    def test_ni_gains(self, nidq_files):
        """NI gains are not in the minimal fixture, but check they don't error."""
        meta_path, _, _, _ = nidq_files
        info = header(meta_path)
        # The example meta doesn't have niMNGain/niMAGain (stripped in fixture)
        # but the field should not be present rather than erroring
        assert info["stream_type"] == "nidq"


class TestHeaderAP:
    def test_stream_type(self, ap_files):
        meta_path, _, _, _ = ap_files
        info = header(meta_path)
        assert info["stream_type"] == "ap"

    def test_sample_rate(self, ap_files):
        meta_path, _, _, _ = ap_files
        info = header(meta_path)
        assert info["sample_rate"] == 30000.0

    def test_channel_counts(self, ap_files):
        meta_path, _, _, _ = ap_files
        info = header(meta_path)
        assert info["n_neural_chans"] == 384
        assert info["n_sync_chans"] == 1
        assert info["n_saved_chans"] == 385

    def test_digital_lines(self, ap_files):
        meta_path, _, _, _ = ap_files
        info = header(meta_path)
        assert info["n_digital_lines"] == 16  # 16 bits * 1 sync chan
        assert info["n_digital_word_cols"] == 1
        assert info["digital_line_label"][0] == "SY0.0"
        assert info["digital_line_label"][6] == "SY0.6"  # SMA sync input

    def test_voltage_range(self, ap_files):
        meta_path, _, _, _ = ap_files
        info = header(meta_path)
        assert info["voltage_range"] == (-0.6, 0.6)


# ---------------------------------------------------------------------------
# Reader: filenamefromepochfiles tests
# ---------------------------------------------------------------------------


class TestFilenamefromepochfiles:
    def test_nidq_bin_derives_meta(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        result = reader.filenamefromepochfiles([bin_path, meta_path])
        assert result == meta_path

    def test_nidq_bin_only(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        # .bin file present, .meta on disk but not in list
        result = reader.filenamefromepochfiles([bin_path])
        assert result.endswith(".nidq.meta")

    def test_ap_meta_only(self, ap_files):
        meta_path, _, _, _ = ap_files
        reader = ndr_reader_neuropixelsGLX()
        result = reader.filenamefromepochfiles([meta_path])
        assert result == meta_path

    def test_ap_bin_derives_meta(self, ap_files):
        meta_path, bin_path, _, _ = ap_files
        reader = ndr_reader_neuropixelsGLX()
        result = reader.filenamefromepochfiles([bin_path, meta_path])
        assert result == meta_path

    def test_no_files_raises(self):
        reader = ndr_reader_neuropixelsGLX()
        with pytest.raises(FileNotFoundError, match="No .meta file"):
            reader.filenamefromepochfiles(["/path/to/data.txt"])

    def test_multiple_meta_no_bin_raises(self, tmp_path):
        reader = ndr_reader_neuropixelsGLX()
        m1 = str(tmp_path / "a.ap.meta")
        m2 = str(tmp_path / "b.nidq.meta")
        Path(m1).touch()
        Path(m2).touch()
        with pytest.raises(ValueError, match="Multiple .meta files"):
            reader.filenamefromepochfiles([m1, m2])

    def test_bin_with_missing_meta_raises(self, tmp_path):
        reader = ndr_reader_neuropixelsGLX()
        binf = str(tmp_path / "missing.nidq.bin")
        Path(binf).touch()
        with pytest.raises(FileNotFoundError, match="No companion .meta"):
            reader.filenamefromepochfiles([binf])


# ---------------------------------------------------------------------------
# Reader: getchannelsepoch tests
# ---------------------------------------------------------------------------


class TestGetchannelsepochNIDQ:
    def test_channel_count(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        channels = reader.getchannelsepoch([bin_path, meta_path])
        # 1 time + 8 analog + 8 digital = 17
        assert len(channels) == 17

    def test_channel_types(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        channels = reader.getchannelsepoch([bin_path, meta_path])
        types = [ch["type"] for ch in channels]
        assert types.count("time") == 1
        assert types.count("analog_in") == 8
        assert types.count("digital_in") == 8

    def test_channel_names(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        channels = reader.getchannelsepoch([bin_path, meta_path])
        names = [ch["name"] for ch in channels]
        assert names[0] == "t1"
        assert names[1] == "ai1"
        assert names[8] == "ai8"
        assert names[9] == "di1"
        assert names[16] == "di8"


class TestGetchannelsepochAP:
    def test_channel_count(self, ap_files):
        meta_path, bin_path, _, _ = ap_files
        reader = ndr_reader_neuropixelsGLX()
        channels = reader.getchannelsepoch([bin_path, meta_path])
        # 1 time + 384 analog + 16 digital = 401
        assert len(channels) == 401

    def test_channel_types(self, ap_files):
        meta_path, bin_path, _, _ = ap_files
        reader = ndr_reader_neuropixelsGLX()
        channels = reader.getchannelsepoch([bin_path, meta_path])
        types = [ch["type"] for ch in channels]
        assert types.count("time") == 1
        assert types.count("analog_in") == 384
        assert types.count("digital_in") == 16


# ---------------------------------------------------------------------------
# Reader: samplerate tests
# ---------------------------------------------------------------------------


class TestSamplerate:
    def test_nidq_samplerate(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        sr = reader.samplerate([bin_path, meta_path], 1, "analog_in", 1)
        assert sr == pytest.approx(10593.220339)

    def test_ap_samplerate(self, ap_files):
        meta_path, bin_path, _, _ = ap_files
        reader = ndr_reader_neuropixelsGLX()
        sr = reader.samplerate([bin_path, meta_path], 1, "analog_in", 1)
        assert sr == 30000.0

    def test_samplerate_array(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        sr = reader.samplerate([bin_path, meta_path], 1, "analog_in", [1, 2, 3])
        assert len(sr) == 3
        np.testing.assert_allclose(sr, 10593.220339)


# ---------------------------------------------------------------------------
# Reader: t0_t1 tests
# ---------------------------------------------------------------------------


class TestT0T1:
    def test_nidq_t0_t1(self, nidq_files):
        meta_path, bin_path, n_chans, n_samples = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        t0t1 = reader.t0_t1([bin_path, meta_path])
        assert len(t0t1) == 1
        assert t0t1[0][0] == 0.0
        expected_t_end = (n_samples - 1) / 10593.220339
        assert t0t1[0][1] == pytest.approx(expected_t_end, rel=1e-6)


# ---------------------------------------------------------------------------
# Reader: readchannels_epochsamples tests
# ---------------------------------------------------------------------------


class TestReadchannelsNIDQ:
    def test_read_analog(self, nidq_files):
        meta_path, bin_path, _, n_samples = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("analog_in", [1], files, 1, 1, 10)
        assert data.shape == (10, 1)
        assert data.dtype == np.int16
        # First analog channel was filled with arange(1, n_samples+1)*1
        np.testing.assert_array_equal(data[:, 0], np.arange(1, 11, dtype=np.int16))

    def test_read_multiple_analog(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("analog_in", [1, 3], files, 1, 1, 5)
        assert data.shape == (5, 2)
        # Channel 1: arange*1, Channel 3: arange*3
        np.testing.assert_array_equal(data[:, 0], np.arange(1, 6, dtype=np.int16))
        np.testing.assert_array_equal(data[:, 1], np.arange(1, 6, dtype=np.int16) * 3)

    def test_read_digital_bit0(self, nidq_files):
        """Bit 0 is set on odd samples (0-indexed: s=1,3,5,...)."""
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        # Read samples 1-8 (1-based)
        data = reader.readchannels_epochsamples("digital_in", [1], files, 1, 1, 8)
        assert data.shape == (8, 1)
        # 0-indexed samples 0..7 → bit0 set at odd indices (1,3,5,7)
        # 1-based samples 1..8 → file rows 0..7
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int16)
        np.testing.assert_array_equal(data[:, 0], expected)

    def test_read_digital_bit1(self, nidq_files):
        """Bit 1 is set on every-4th sample (0-indexed: s=0,4,8,...)."""
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("digital_in", [2], files, 1, 1, 8)
        expected = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int16)
        np.testing.assert_array_equal(data[:, 0], expected)

    def test_read_digital_bit2(self, nidq_files):
        """Bit 2 is set on every-8th sample (0-indexed: s=0,8,...)."""
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("digital_in", [3], files, 1, 1, 8)
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16)
        np.testing.assert_array_equal(data[:, 0], expected)

    def test_read_digital_multiple_lines(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("digital_in", [1, 2, 3], files, 1, 1, 4)
        assert data.shape == (4, 3)
        # bit 0 at samples 1-4: [0,1,0,1]
        np.testing.assert_array_equal(data[:, 0], [0, 1, 0, 1])
        # bit 1 at samples 1-4: [1,0,0,0]
        np.testing.assert_array_equal(data[:, 1], [1, 0, 0, 0])
        # bit 2 at samples 1-4: [1,0,0,0]
        np.testing.assert_array_equal(data[:, 2], [1, 0, 0, 0])

    def test_read_digital_out_of_range(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        with pytest.raises(ValueError, match="Digital line out of range"):
            reader.readchannels_epochsamples("digital_in", [9], files, 1, 1, 5)

    def test_read_time(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("time", 1, files, 1, 1, 3)
        assert data.shape[0] == 3
        assert data.dtype == np.float64


class TestReadchannelsAP:
    def test_read_analog(self, ap_files):
        meta_path, bin_path, _, _ = ap_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("analog_in", [1], files, 1, 1, 5)
        assert data.shape == (5, 1)
        assert data.dtype == np.int16

    def test_read_digital_sync_bit6(self, ap_files):
        """For AP, digital_in di7 = bit 6 of the sync word (SMA sync input)."""
        meta_path, bin_path, _, _ = ap_files
        reader = ndr_reader_neuropixelsGLX()
        files = [bin_path, meta_path]
        data = reader.readchannels_epochsamples("digital_in", [7], files, 1, 1, 5)
        assert data.shape == (5, 1)
        assert data.dtype == np.int16


# ---------------------------------------------------------------------------
# Reader: epochclock tests
# ---------------------------------------------------------------------------


class TestEpochclock:
    def test_returns_dev_local_time(self, nidq_files):
        meta_path, bin_path, _, _ = nidq_files
        reader = ndr_reader_neuropixelsGLX()
        ec = reader.epochclock([bin_path, meta_path])
        assert len(ec) == 1
        assert ec[0].type == "dev_local_time"
