"""Read digital channel data from a SpikeGadgets .rec file.

Port of +ndr/+format/+spikegadgets/read_rec_digitalChannels.m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ndr.format.spikegadgets.read_rec_configsize import read_rec_configsize


def read_rec_digitalChannels(
    filename: str | Path,
    NumChannels: str | int,
    channels: np.ndarray | list,
    samplingRate: float,
    headerSize: str | int,
    s0: int,
    s1: int,
    configExists: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read digital channel data from a SpikeGadgets raw file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.rec`` file.
    NumChannels : str or int
        Number of channels.
    channels : array-like
        N x 2 array where each row is ``[byte_position(1-based), bit_position(1-based)]``.
    samplingRate : float
        Sampling rate in Hz.
    headerSize : str or int
        Header size in int16 units.
    s0 : int
        First sample (1-based).
    s1 : int
        Last sample (1-based).
    configExists : bool
        Whether the file has an XML configuration header.

    Returns
    -------
    recData : ndarray
        M x N boolean array (M channels, N samples).
    timestamps : ndarray
        Timestamps in seconds.
    """
    filename = Path(filename)
    channels = np.asarray(channels)
    num_channels = int(NumChannels)
    header_size_bytes = int(headerSize) * 2
    channel_size_bytes = num_channels * 2
    # One sample occupies: [header][4-byte uint32 timestamp][channel data].
    # MATLAB's `blockSizeBytes` (headerSizeBytes + 2 + channelSizeBytes) is NOT
    # this stride -- it is the skip argument to fread, which advances *after*
    # reading, so MATLAB's true stride is header + 2(read) + 2 + channel =
    # header + 4 + channel. Porting the MATLAB expression verbatim as a total
    # stride left every read 2 bytes short per sample, drifting steadily off
    # the correct offset (and, since the drift is not a multiple of the
    # per-channel stride, onto the wrong channel).
    block_size_bytes = header_size_bytes + 4 + channel_size_bytes

    # MATLAB's strfind is 1-based, so its `+ 16` lands on the first packet
    # byte; this port applied the same `+ 16` to 0-based bytes.find and
    # landed on the line terminator, shifting every read by one byte.
    # See read_rec_configsize.
    configsize = read_rec_configsize(filename) if configExists else 0

    num_samples = s1 - s0 + 1

    # Read timestamps
    with open(filename, "rb") as f:
        f.seek(configsize + header_size_bytes)
        f.seek((s0 - 1) * block_size_bytes, 1)
        timestamps = np.empty(num_samples, dtype=np.float64)
        for i in range(num_samples):
            ts_bytes = f.read(4)
            if len(ts_bytes) < 4:
                raise EOFError(
                    f"Requested samples {s0}..{s1} but {filename} holds only {i} "
                    "timestamps from that offset."
                )
            timestamps[i] = np.frombuffer(ts_bytes, dtype=np.dtype("<u4"))[0]
            if i < num_samples - 1:
                f.seek(header_size_bytes + channel_size_bytes, 1)
        timestamps /= samplingRate

    # Read digital data
    bytes_to_read = np.unique(channels[:, 0])
    rec_data_list = []

    with open(filename, "rb") as f:
        for byte_loc in bytes_to_read:
            f.seek(configsize + int(byte_loc) - 1)
            f.seek((s0 - 1) * block_size_bytes, 1)
            tmp = np.empty(num_samples, dtype=np.uint8)
            # One byte is read per sample, so the skip is the stride less
            # that byte: block_size_bytes - 1. Spelled out to mirror the
            # MATLAB fread skip argument (headerSizeBytes+3+channelSizeBytes).
            skip = block_size_bytes - 1
            for j in range(num_samples):
                raw = f.read(1)
                if len(raw) < 1:
                    # tmp came from np.empty; the bits unpacked from its
                    # uninitialized tail would look like real digital states.
                    raise EOFError(
                        f"Requested samples {s0}..{s1} but {filename} holds only "
                        f"{j} samples for the byte at location {int(byte_loc)}."
                    )
                tmp[j] = raw[0]
                if j < num_samples - 1:
                    f.seek(skip, 1)

            # Extract bits for channels sharing this byte
            current_channels = np.where(channels[:, 0] == byte_loc)[0]
            for ch_idx in current_channels:
                bit = int(channels[ch_idx, 1]) - 1  # convert to 0-based
                rec_data_list.append(((tmp >> bit) & 1).astype(bool))

    rec_data = np.array(rec_data_list) if rec_data_list else np.empty((0, num_samples), dtype=bool)

    return rec_data, timestamps
