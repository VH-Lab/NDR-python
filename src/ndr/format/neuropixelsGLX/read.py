"""Read data from a SpikeGLX Neuropixels binary (.bin) file.

Port of +ndr/+format/+neuropixelsGLX/read.m
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ndr.format.binarymatrix.read import read as binarymatrix_read
from ndr.format.neuropixelsGLX.header import header
from ndr.time.fun.samples2times import samples2times
from ndr.time.fun.times2samples import times2samples


def read(
    binfilename: str | Path,
    t0: float,
    t1: float,
    *,
    numChans: int = 0,
    SR: float = 0.0,
    channels: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Read data from a SpikeGLX Neuropixels binary (.bin) file.

    Parameters
    ----------
    binfilename : str or Path
        Full path to the .bin file.
    t0 : float
        Start time in seconds. Use ``-inf`` to start from the beginning.
    t1 : float
        End time in seconds. Use ``inf`` to read to the end.
    numChans : int, optional
        Total number of channels in the file including sync. If 0, read
        from companion .meta file.
    SR : float, optional
        Sampling rate in Hz. If 0, read from companion .meta file.
    channels : array-like of int, optional
        1-based channel indices to read. ``None`` means all channels.

    Returns
    -------
    data : ndarray
        N x C int16 matrix of samples.
    t : ndarray
        N x 1 time vector in seconds.
    t0_t1 : list of float
        [start_time, end_time] for the full file.
    """
    binfilename = Path(binfilename)

    # If numChans or SR not provided, read from companion .meta file
    if numChans == 0 or SR == 0.0:
        metafilename = binfilename.with_suffix(".meta")
        if not metafilename.is_file():
            # Try replacing .bin with .meta in the full name
            meta_str = str(binfilename)[:-3] + "meta"
            metafilename = Path(meta_str)
        if not metafilename.is_file():
            raise FileNotFoundError(
                f"No .meta file found at {metafilename} and numChans/SR not specified."
            )
        info = header(str(metafilename))
        if numChans == 0:
            numChans = info["n_saved_chans"]
        if SR == 0.0:
            SR = info["sample_rate"]

    bytes_per_sample = 2  # int16
    bytes_per_full_sample = bytes_per_sample * numChans

    # Validate channels
    if channels is not None:
        channels_arr = np.asarray(channels, dtype=np.int64).ravel()
        if np.any(channels_arr > numChans) or np.any(channels_arr < 1):
            raise ValueError(f"Channel indices must be between 1 and {numChans}.")
    else:
        channels_arr = np.arange(1, numChans + 1, dtype=np.int64)

    # Calculate total time range (no header in SpikeGLX .bin files)
    filesize = binfilename.stat().st_size
    total_samples = filesize // bytes_per_full_sample

    file_t_start = 0.0
    file_t_end = float(samples2times(np.array([total_samples]), [0, 0], SR)[0])
    t0_t1 = [file_t_start, file_t_end]

    # Clamp requested times
    t0_req = max(t0, t0_t1[0]) if not (math.isinf(t0) and t0 < 0) else t0_t1[0]
    t1_req = min(t1, t0_t1[1]) if not (math.isinf(t1) and t1 > 0) else t0_t1[1]

    if t1_req < t0_req:
        return np.array([], dtype=np.int16), np.array([]), t0_t1

    # Convert times to samples
    s0_req = int(times2samples(np.array([t0_req]), t0_t1, SR)[0])
    s1_req = int(times2samples(np.array([t1_req]), t0_t1, SR)[0])
    s0_req = max(1, s0_req)
    s1_req = min(total_samples, s1_req)

    if s1_req < s0_req:
        return np.array([], dtype=np.int16), np.array([]), t0_t1

    # Read using binarymatrix
    data, _, s0_actual, s1_actual = binarymatrix_read(
        binfilename,
        numChans,
        channels_arr,
        float(s0_req),
        float(s1_req),
        dataType="int16",
        byteOrder="ieee-le",
        headerSkip=0,
    )

    # Generate time vector
    if data.size > 0:
        t = samples2times(
            np.arange(s0_actual, s1_actual + 1, dtype=float), t0_t1, SR
        )
    else:
        t = np.array([])
        data = np.array([], dtype=np.int16)

    return data, t, t0_t1
