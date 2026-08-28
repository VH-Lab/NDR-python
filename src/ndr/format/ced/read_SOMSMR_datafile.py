"""Read samples from a CED SOM/SMR file.

Port of +ndr/+format/+ced/read_SOMSMR_datafile.m

This module requires the ``neo`` package for reading CED files.
The MATLAB version depends heavily on sigTOOL (SONGetBlockHeaders,
SONGetADCChannel, etc.). This Python port uses the Neo library as the
backend reader, which provides equivalent CED reading capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import neo
    from neo.rawio.cedrawio import CedRawIO
except ImportError:  # pragma: no cover
    neo = None  # type: ignore[assignment]
    CedRawIO = None  # type: ignore[assignment]

from ndr.format.ced.read_SOMSMR_header import read_SOMSMR_header
from ndr.time.fun.times2samples import matlab_round


def read_SOMSMR_datafile(
    filename: str | Path,
    header: dict[str, Any] | None = None,
    channel_number: int = 1,
    t0: float = 0.0,
    t1: float = float("inf"),
) -> tuple[np.ndarray, int | None, float | None, Any, np.ndarray | None]:
    """Read samples from a CED SOM/SMR file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.smr`` or ``.son`` file.
    header : dict or None
        Header information. If None, it will be read from the file.
    channel_number : int
        Channel number to read (corresponds to Spike2 channel number).
    t0 : float
        Start time in seconds.
    t1 : float
        End time in seconds.

    Returns
    -------
    data : ndarray
        Data samples (column vector).
    total_samples : int or None
        Total number of samples in the channel.
    total_time : float or None
        Total duration in seconds.
    blockinfo : object
        Block information (Neo raw reader for further queries).
    time : ndarray or None
        Time vector for each sample.
    """
    if neo is None:
        raise ImportError("neo is required for reading CED files. Install with: pip install neo")

    filename = Path(filename)

    if header is None:
        header = read_SOMSMR_header(filename)

    raw_reader = CedRawIO(filename=str(filename))
    raw_reader.parse_header()

    sig_channels = raw_reader.header["signal_channels"]
    ch_ids = sig_channels["id"]

    # Find the requested channel
    ch_idx = None
    for idx, cid in enumerate(ch_ids):
        if int(cid) == channel_number:
            ch_idx = idx
            break

    if ch_idx is None:
        raise ValueError(f"Channel number {channel_number} not recorded in file {filename}.")

    # Resolve the channel's stream BEFORE sizing anything. A Spike2 file mixes
    # sample rates, and neo groups equally-sampled channels into streams, so
    # the length, the rate and the sample clamp must all come from the
    # requested channel's own stream. get_signal_size with no stream_index
    # returns stream 0's length, which was then paired with this channel's
    # rate -- a nonsense duration for any channel outside stream 0.
    stream_id = sig_channels[ch_idx]["stream_id"]
    streams = raw_reader.header["signal_streams"]
    stream_idx = 0
    for si, st in enumerate(streams):
        if st["id"] == stream_id:
            stream_idx = si
            break

    # get_analogsignal_chunk indexes channels within the stream, not within the
    # flat signal_channels array. Passing the flat index reads whichever channel
    # happens to sit at that position in the stream.
    stream_channel_positions = np.nonzero(sig_channels["stream_id"] == stream_id)[0]
    stream_channel_index = int(np.nonzero(stream_channel_positions == ch_idx)[0][0])

    sr = float(sig_channels[ch_idx]["sampling_rate"])
    n_samples = raw_reader.get_signal_size(
        block_index=0, seg_index=0, stream_index=stream_idx
    )
    total_samples = n_samples
    total_time = n_samples / sr

    if t0 < 0:
        t0 = 0.0
    if t1 > total_time:
        t1 = total_time

    s0 = max(0, int(matlab_round(t0 * sr)))
    s1 = min(n_samples, int(matlab_round(t1 * sr)) + 1)

    raw = raw_reader.get_analogsignal_chunk(
        block_index=0,
        seg_index=0,
        i_start=s0,
        i_stop=s1,
        channel_indexes=[stream_channel_index],
        stream_index=stream_idx,
    )
    data = raw_reader.rescale_signal_raw_to_float(
        raw, channel_indexes=[stream_channel_index], stream_index=stream_idx
    )
    data = data.ravel()

    time = np.arange(s0, s0 + len(data)) / sr

    return data.reshape(-1, 1), total_samples, total_time, raw_reader, time.reshape(-1, 1)
