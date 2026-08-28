"""Pin where SpikeGadgets packet data begins in a real .rec file.

A .rec starts with an XML configuration block; the packet stream begins on the
line AFTER ``</Configuration>``. Taking the offset as the end of the tag leaves
its line terminator unconsumed, so every read is shifted one byte and each
int16 sample is assembled from the high byte of one sample and the low byte of
the next.

That misalignment cannot be caught by shape or range checks -- it yields
plausible-looking int16s. It is caught here two ways against example.rec:
a hardware sample counter that must increment by exactly 1, and the fact that
an electrophysiology trace is smooth sample-to-sample while a byte-shifted one
is not.
"""

import numpy as np
import pytest

from ndr.format.spikegadgets.read_rec_analogChannels import read_rec_analogChannels
from ndr.format.spikegadgets.read_rec_config import read_rec_config
from ndr.format.spikegadgets.read_rec_configsize import read_rec_configsize
from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels
from ndr.fun.ndrpath import ndrpath

REC = ndrpath() / "example_data" / "example.rec"
TAG = b"</Configuration>"


@pytest.fixture(scope="module")
def rec():
    if not REC.exists():
        pytest.skip("example.rec not available")
    return REC


@pytest.fixture(scope="module")
def geometry(rec):
    cfg, _channels = read_rec_config(str(rec))
    num_channels = int(cfg["numChannels"])
    header_size = int(cfg["headerSize"])
    header_bytes = header_size * 2
    return {
        "num_channels": num_channels,
        "header_size": header_size,
        "header_bytes": header_bytes,
        # [header][4-byte timestamp][channel data]
        "stride": header_bytes + 4 + num_channels * 2,
    }


class TestConfigSize:
    def test_offset_lands_past_the_line_terminator(self, rec):
        raw = rec.read_bytes()
        tag_end = raw.find(TAG) + len(TAG)
        assert read_rec_configsize(rec) == tag_end + 1
        assert raw[tag_end : tag_end + 1] == b"\n", "this fixture ends the header with \\n"

    def test_missing_config_block_returns_zero(self, tmp_path):
        f = tmp_path / "noconfig.rec"
        f.write_bytes(b"\x00\x01\x02\x03" * 100)
        assert read_rec_configsize(f) == 0

    @pytest.mark.parametrize(
        "terminator,expected_extra",
        [(b"\n", 1), (b"\r\n", 2), (b"\r", 1), (b"", 0), (b"\x00", 0)],
    )
    def test_terminator_forms(self, tmp_path, terminator, expected_extra):
        """A file written without a trailing newline must still resolve."""
        f = tmp_path / "x.rec"
        head = b"<Configuration>stuff" + TAG
        f.write_bytes(head + terminator + b"\x01\x02\x03\x04")
        assert read_rec_configsize(f) == len(head) + expected_extra


class TestAlignmentAgainstRealData:
    def test_timestamps_increment_by_exactly_one(self, rec, geometry):
        """The decisive check: a hardware sample counter has no gaps.

        Misaligned by a byte, this field is noise; misaligned by the old
        276-byte stride, likewise.
        """
        raw = rec.read_bytes()
        start = read_rec_configsize(rec) + geometry["header_bytes"]
        stride = geometry["stride"]
        ts = np.array(
            [int.from_bytes(raw[start + i * stride :][:4], "little") for i in range(5000)],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(np.diff(ts), 1)

    def test_a_one_byte_shift_destroys_the_timestamps(self, rec, geometry):
        """Guard the guard: prove the check above can actually fail."""
        raw = rec.read_bytes()
        stride = geometry["stride"]
        start = read_rec_configsize(rec) + geometry["header_bytes"] - 1  # the old offset
        ts = np.array(
            [int.from_bytes(raw[start + i * stride :][:4], "little") for i in range(500)],
            dtype=np.int64,
        )
        assert not np.all(np.diff(ts) == 1)

    def test_trode_trace_is_smooth(self, rec, geometry):
        """Ephys is continuous; a byte-shifted read is not.

        mean|diff| / std is far below 1 for a real trace and near or above it
        for noise, which is what the misaligned read produced.
        """
        cfg, _c = read_rec_config(str(rec))
        data, _ts = read_rec_trodeChannels(
            rec, geometry["num_channels"], [1], 30000.0, geometry["header_size"], 1, 4000
        )
        trace = np.asarray(data).ravel().astype(float)
        assert np.abs(np.diff(trace)).mean() / trace.std() < 0.25

    def test_analog_timestamps_are_uniformly_spaced(self, rec, geometry):
        """The reader's own timestamp output, not just the raw bytes."""
        _data, ts = read_rec_analogChannels(
            rec, geometry["num_channels"], [1], 30000.0, geometry["header_size"], 1, 2000, True
        )
        steps = np.diff(np.asarray(ts, dtype=np.float64))
        # Tolerance is set by float64, not by the data: these timestamps are
        # ~2028 s, where one ULP is ~4.5e-13 against a 3.3e-5 step, so
        # consecutive differences wobble in the last bits. A misalignment is
        # nowhere near this small -- it scrambles the counter entirely.
        np.testing.assert_allclose(steps, 1.0 / 30000.0, atol=1e-11)
