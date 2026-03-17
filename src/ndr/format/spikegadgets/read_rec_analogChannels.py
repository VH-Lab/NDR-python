"""Read analog channel data from a SpikeGadgets .rec file.

Port of +ndr/+format/+spikegadgets/read_rec_analogChannels.m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_rec_analogChannels(
    filename: str | Path,
    NumChannels: str | int,
    channels: list[int],
    samplingRate: float,
    headerSize: str | int,
    s0: int,
    s1: int,
    configExists: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read analog channel data from a SpikeGadgets raw file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.rec`` file.
    NumChannels : str or int
        Number of channels.
    channels : list of int
        1-based byte locations of analog channels to read.
    samplingRate : float
        Sampling rate in Hz.
    headerSize : str or int
        Header size in int16 units.
    s0 : int
        First sample to read (1-based).
    s1 : int
        Last sample to read (1-based).
    configExists : bool
        Whether the file has an XML configuration header.

    Returns
    -------
    recData : ndarray
        M x N matrix (M channels, N samples).
    timestamps : ndarray
        Timestamps in seconds.
    """
    filename = Path(filename)
    num_channels = int(NumChannels)
    header_size_bytes = int(headerSize) * 2
    channel_size_bytes = num_channels * 2
    block_size_bytes = header_size_bytes + 2 + channel_size_bytes

    configsize = 0
    if configExists:
        with open(filename, "rb") as f:
            junk = f.read(30000)
        config_end = junk.find(b"</Configuration>")
        configsize = config_end + 16 if config_end >= 0 else 0

    num_samples = s1 - s0 + 1

    # Read timestamps
    with open(filename, "rb") as f:
        f.seek(configsize + header_size_bytes)
        timestamps = np.empty(num_samples, dtype=np.float64)
        f.seek((s0 - 1) * block_size_bytes, 1)
        for i in range(num_samples):
            ts_bytes = f.read(4)
            if len(ts_bytes) < 4:
                timestamps = timestamps[:i]
                break
            timestamps[i] = np.frombuffer(ts_bytes, dtype=np.dtype("<u4"))[0]
            if i < num_samples - 1:
                f.seek(header_size_bytes + channel_size_bytes, 1)
        timestamps /= samplingRate

    # Read analog data
    rec_data = np.empty((len(channels), num_samples), dtype=np.int16)
    with open(filename, "rb") as f:
        for i, byte_loc in enumerate(channels):
            f.seek(configsize + byte_loc - 1)
            f.seek((s0 - 1) * block_size_bytes, 1)
            for j in range(num_samples):
                raw = f.read(2)
                if len(raw) < 2:
                    break
                rec_data[i, j] = np.frombuffer(raw, dtype=np.dtype("<i2"))[0]
                if j < num_samples - 1:
                    f.seek(block_size_bytes - 2, 1)

    return rec_data, timestamps
