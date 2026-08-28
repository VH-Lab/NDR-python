"""Pin the SpikeGadgets .rec per-sample stride.

A .rec sample is laid out as::

    [header: headerSize int16s][timestamp: uint32][channel data: NumChannels int16s]

so the stride is ``header + 4 + channel`` bytes. MATLAB's ``blockSizeBytes``
(``headerSizeBytes + 2 + channelSizeBytes``) is the *skip* argument to
``fread``, which advances after the read completes -- it is deliberately 2 less
than the stride for an int16 read. Porting that expression as a total stride
under-advanced every read by 2 bytes per sample.

These tests use a synthetic file with a known value at every position, so an
off-by-two shows up as wrong data rather than as a plausible-looking waveform.
The bundled example.rec cannot do that: any stride produces *some* numbers.
"""

import struct

import numpy as np
import pytest

from ndr.format.spikegadgets.read_rec_analogChannels import read_rec_analogChannels
from ndr.format.spikegadgets.read_rec_digitalChannels import read_rec_digitalChannels
from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels

HEADER_INT16S = 4
NUM_CHANNELS = 3
N_SAMPLES = 10
SAMPLING_RATE = 30000.0

HEADER_BYTES = HEADER_INT16S * 2
CHANNEL_BYTES = NUM_CHANNELS * 2
STRIDE = HEADER_BYTES + 4 + CHANNEL_BYTES


def trode_value(sample, channel):
    """Value stored for a 1-based channel at a 0-based sample."""
    return sample * 10 + channel


def header_value(sample, slot):
    """Value stored in the 1-based header int16 slot at a 0-based sample."""
    return 5000 + sample * 10 + slot


@pytest.fixture
def recfile(tmp_path):
    """A .rec with no config block and a distinct value at every position."""
    path = tmp_path / "synthetic.rec"
    with open(path, "wb") as f:
        for s in range(N_SAMPLES):
            for slot in range(1, HEADER_INT16S + 1):
                f.write(struct.pack("<h", header_value(s, slot)))
            f.write(struct.pack("<I", 1000 + s))
            for c in range(1, NUM_CHANNELS + 1):
                f.write(struct.pack("<h", trode_value(s, c)))
    assert path.stat().st_size == N_SAMPLES * STRIDE
    return path


class TestTrodeStride:
    def test_reads_every_sample_from_the_start(self, recfile):
        data, ts = read_rec_trodeChannels(
            recfile, NUM_CHANNELS, [1, 3], SAMPLING_RATE, HEADER_INT16S, 1, N_SAMPLES
        )
        scale = 12780.0 / 65536.0
        expected = np.array(
            [[trode_value(s, 1) * scale, trode_value(s, 3) * scale] for s in range(N_SAMPLES)]
        )
        np.testing.assert_allclose(data, expected)
        np.testing.assert_allclose(ts, [(1000 + s) / SAMPLING_RATE for s in range(N_SAMPLES)])

    def test_s0_offset_lands_on_the_right_sample(self, recfile):
        """The off-by-two only shows up once s0 > 1: it scales with s0."""
        data, _ = read_rec_trodeChannels(
            recfile, NUM_CHANNELS, [2], SAMPLING_RATE, HEADER_INT16S, 4, 6
        )
        scale = 12780.0 / 65536.0
        expected = np.array([[trode_value(s, 2) * scale] for s in (3, 4, 5)])
        np.testing.assert_allclose(data, expected)

    def test_reading_past_the_end_raises(self, recfile):
        with pytest.raises(EOFError, match="holds only"):
            read_rec_trodeChannels(
                recfile, NUM_CHANNELS, [1], SAMPLING_RATE, HEADER_INT16S, 1, N_SAMPLES + 5
            )


class TestAnalogStride:
    def test_header_slot_reads_track_the_sample(self, recfile):
        """Analog channels are addressed by byte location inside the header."""
        data, ts = read_rec_analogChannels(
            recfile, NUM_CHANNELS, [1, 5], SAMPLING_RATE, HEADER_INT16S, 1, N_SAMPLES, False
        )
        expected = np.array(
            [
                [header_value(s, 1) for s in range(N_SAMPLES)],
                [header_value(s, 3) for s in range(N_SAMPLES)],
            ]
        )
        np.testing.assert_array_equal(data, expected)
        np.testing.assert_allclose(ts, [(1000 + s) / SAMPLING_RATE for s in range(N_SAMPLES)])

    def test_s0_offset_lands_on_the_right_sample(self, recfile):
        data, _ = read_rec_analogChannels(
            recfile, NUM_CHANNELS, [1], SAMPLING_RATE, HEADER_INT16S, 4, 6, False
        )
        np.testing.assert_array_equal(data, np.array([[header_value(s, 1) for s in (3, 4, 5)]]))

    def test_reading_past_the_end_raises(self, recfile):
        with pytest.raises(EOFError, match="holds only"):
            read_rec_analogChannels(
                recfile, NUM_CHANNELS, [1], SAMPLING_RATE, HEADER_INT16S, 1, N_SAMPLES + 5, False
            )


class TestDigitalStride:
    def test_bits_track_the_sample(self, recfile):
        """Byte 1 of the header is the low byte of header_value(s, 1)."""
        channels = np.array([[1, 1], [1, 2]])
        data, _ = read_rec_digitalChannels(
            recfile, NUM_CHANNELS, channels, SAMPLING_RATE, HEADER_INT16S, 1, N_SAMPLES, False
        )
        low_bytes = [header_value(s, 1) & 0xFF for s in range(N_SAMPLES)]
        np.testing.assert_array_equal(data[0], [(b >> 0) & 1 for b in low_bytes])
        np.testing.assert_array_equal(data[1], [(b >> 1) & 1 for b in low_bytes])

    def test_reading_past_the_end_raises(self, recfile):
        with pytest.raises(EOFError, match="holds only"):
            read_rec_digitalChannels(
                recfile,
                NUM_CHANNELS,
                np.array([[1, 1]]),
                SAMPLING_RATE,
                HEADER_INT16S,
                1,
                N_SAMPLES + 5,
                False,
            )
