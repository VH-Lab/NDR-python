"""Read data from a Dabrowska lab MAT file.

Port of +ndr/+format/+dabrowska/read.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover
    sio = None  # type: ignore[assignment]

from ndr.format.dabrowska.header import header as read_header
from ndr.time.fun.times2samples import times2samples


def read(
    filename: str | Path,
    channeltype: str = "ai",
    header: dict[str, Any] | None = None,
) -> np.ndarray:
    """Read data from a Dabrowska lab MAT file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.mat`` file.
    channeltype : str
        ``'time'`` / ``'t'`` for time vector, ``'analog_in'`` / ``'ai'``
        for input data, ``'analog_out'`` / ``'ao'`` for output data.
    header : dict or None
        Pre-loaded header. If None, reads from file.

    Returns
    -------
    numpy.ndarray
        Column vector of the requested data.
    """
    if sio is None:
        raise ImportError("scipy is required for reading MAT files.")

    filename = Path(filename)
    if filename.suffix != ".mat":
        raise ValueError(f'Expected file "{filename}" to have a ".mat" extension.')

    if header is None or len(header) == 0:
        header = read_header(filename)

    mat = sio.loadmat(str(filename), squeeze_me=True)

    trigger_times = np.asarray(header["triggerTime"], dtype=np.float64).ravel()
    duration_ms = float(header["duration"])
    sample_rate = float(header["sampleRate"])
    number_steps = len(trigger_times)

    # Convert MATLAB datenum to relative seconds
    # t0_global_steps in seconds relative to earliest trigger
    t0_seconds = (trigger_times - trigger_times.min()) * 86400.0  # datenum is in days
    t1_seconds = t0_seconds + duration_ms / 1000.0

    t0_local = float(t0_seconds.min())
    t1_local = float(t1_seconds.max())

    # Compute step sample indices
    step_ind = np.zeros((number_steps, 2), dtype=int)
    step_ind[:, 0] = times2samples(t0_seconds, [t0_local, t1_local], sample_rate).astype(int)
    step_ind[:, 1] = times2samples(t1_seconds, [t0_local, t1_local], sample_rate).astype(int)

    num_samples = int(step_ind[-1, 1])

    ct = channeltype.lower()

    if ct in ("time", "t"):
        data = np.linspace(t0_local, t1_local, num_samples)
    elif ct in ("analog_in", "ai"):
        input_data = np.asarray(mat["inputData"], dtype=np.float64)
        data = np.full(num_samples, np.nan)
        for i in range(number_steps):
            idx = slice(step_ind[i, 0] - 1, step_ind[i, 1] - 1)
            step_data = input_data[:, i] if input_data.ndim == 2 else input_data
            data[idx] = step_data.ravel()[: idx.stop - idx.start]
    elif ct in ("analog_out", "ao"):
        output_data = np.asarray(mat["outputData"], dtype=np.float64)
        data = np.full(num_samples, np.nan)
        for i in range(number_steps):
            idx = slice(step_ind[i, 0] - 1, step_ind[i, 1] - 1)
            step_data = output_data[:, i] if output_data.ndim == 2 else output_data
            data[idx] = step_data.ravel()[: idx.stop - idx.start]
    else:
        raise ValueError(f'"{channeltype}" is not a valid channel type')

    return data.reshape(-1, 1)
