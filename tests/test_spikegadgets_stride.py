"""Planted-value regression test for the SpikeGadgets per-sample stride.

The true .rec block is header + 4-byte uint32 timestamp + channel data. The
reader previously advanced only header + 2 + channel, drifting 2 bytes per
sample so every sample after the first came from the wrong offset. We build a
synthetic .rec with known planted int16 values and read sample 1 and sample
1000; a wrong stride returns the wrong values for the later sample.
"""

from __future__ import annotations

import struct

import numpy as np

from ndr.format.spikegadgets.read_rec_analogChannels import read_rec_analogChannels

HEADER_SIZE_INT16 = 17  # header size in int16 units
NUM_CHANNELS = 32
NUM_SAMPLES = 1000


def _planted(sample_index: int, channel_index: int) -> int:
    # Distinct per (sample, channel) and safely inside int16 range.
    return sample_index * 30 + channel_index


def _build_synthetic_rec(path) -> None:
    header_size_bytes = HEADER_SIZE_INT16 * 2
    config = b"<Configuration></Configuration>"
    with open(path, "wb") as f:
        f.write(config)
        for n in range(NUM_SAMPLES):
            f.write(b"\x00" * header_size_bytes)  # per-block header
            f.write(struct.pack("<I", n))  # 4-byte uint32 timestamp
            for c in range(NUM_CHANNELS):
                f.write(struct.pack("<h", _planted(n, c)))


def test_analog_stride_reads_exact_planted_values(tmp_path):
    rec = tmp_path / "synthetic.rec"
    _build_synthetic_rec(rec)

    channel_index = 5  # 0-based channel to read
    header_size_bytes = HEADER_SIZE_INT16 * 2
    # byte_loc is the 1-based offset (from block start) of this channel's sample:
    # header + 4-byte timestamp + channel_index * 2, then +1 for 1-based.
    byte_loc = header_size_bytes + 4 + channel_index * 2 + 1

    data, timestamps = read_rec_analogChannels(
        rec,
        NUM_CHANNELS,
        [byte_loc],
        samplingRate=30000.0,
        headerSize=HEADER_SIZE_INT16,
        s0=1,
        s1=NUM_SAMPLES,
        configExists=True,
    )

    assert data.shape == (1, NUM_SAMPLES)
    # Sample 1 was always correct; sample 1000 exposes any stride drift.
    assert data[0, 0] == _planted(0, channel_index)
    assert data[0, NUM_SAMPLES - 1] == _planted(NUM_SAMPLES - 1, channel_index)
    # Every sample must match its planted value.
    expected = np.array([_planted(n, channel_index) for n in range(NUM_SAMPLES)])
    assert np.array_equal(data[0], expected)
    # Timestamps are the block index / sampling rate.
    assert np.isclose(timestamps[0], 0.0)
    assert np.isclose(timestamps[-1], (NUM_SAMPLES - 1) / 30000.0)
