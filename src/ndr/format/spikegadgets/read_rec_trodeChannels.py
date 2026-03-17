"""Read trode channel data from a SpikeGadgets .rec file.

Port of +ndr/+format/+spikegadgets/read_rec_trodeChannels.m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_rec_trodeChannels(
    filename: str | Path,
    NumChannels: str | int,
    channels: list[int],
    samplingRate: float,
    headerSize: str | int,
    s0: int,
    s1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read trode channel data from a SpikeGadgets raw file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.rec`` file.
    NumChannels : str or int
        Number of channels in the recording.
    channels : list of int
        1-based channel indices to read.
    samplingRate : float
        Sampling rate in Hz.
    headerSize : str or int
        Size of the header block in int16 units.
    s0 : int
        First sample to read (1-based).
    s1 : int
        Last sample to read (1-based).

    Returns
    -------
    recData : ndarray
        N x M matrix (N samples, M channels). Scaled to microvolts.
    timestamps : ndarray
        Timestamps in seconds.
    """
    filename = Path(filename)
    num_channels = int(NumChannels)
    header_size_int16 = int(headerSize)

    header_size_bytes = header_size_int16 * 2
    channel_size_bytes = num_channels * 2
    block_size_bytes = header_size_bytes + 2 + channel_size_bytes  # +2 for padding/sync

    # Find config size
    with open(filename, "rb") as f:
        junk = f.read(30000)
    config_end = junk.find(b"</Configuration>")
    configsize = config_end + 16 if config_end >= 0 else 0

    # Read timestamps
    with open(filename, "rb") as f:
        f.seek(configsize + header_size_bytes)
        num_samples = s1 - s0 + 1
        timestamps = np.empty(num_samples, dtype=np.float64)
        # Skip to s0
        f.seek(configsize + header_size_bytes + (s0 - 1) * block_size_bytes)
        for i in range(num_samples):
            ts_bytes = f.read(4)
            if len(ts_bytes) < 4:
                timestamps = timestamps[:i]
                break
            timestamps[i] = np.frombuffer(ts_bytes, dtype=np.dtype("<u4"))[0]
            f.seek(header_size_bytes + channel_size_bytes, 1)
        timestamps /= samplingRate

    # Read channel data
    rec_data = np.empty((len(channels), s1 - s0 + 1), dtype=np.float64)
    with open(filename, "rb") as f:
        for i, ch in enumerate(channels):
            f.seek(configsize + header_size_bytes + 4 + (ch - 1) * 2)  # +4 for timestamp
            f.seek((s0 - 1) * block_size_bytes, 1)

            col = np.empty(s1 - s0 + 1, dtype=np.int16)
            for j in range(s1 - s0 + 1):
                raw = f.read(2)
                if len(raw) < 2:
                    col = col[:j]
                    break
                col[j] = np.frombuffer(raw, dtype=np.dtype("<i2"))[0]
                if j < s1 - s0:
                    f.seek(block_size_bytes - 2, 1)

            channel_data = col.astype(np.float64)
            channel_data = channel_data * 12780.0 / 65536.0  # convert to µV
            rec_data[i, : len(channel_data)] = channel_data

    rec_data = rec_data.T  # N samples x M channels
    timestamps = timestamps[: rec_data.shape[0]]

    return rec_data, timestamps
