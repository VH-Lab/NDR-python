"""Read data from Intan RHD2000 directory format (one file per channel).

Port of +ndr/+format/+intan/read_Intan_RHD2000_directory.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header


def read_Intan_RHD2000_directory(
    directoryname: str | Path,
    header: dict[str, Any] | None = None,
    channel_type: str = "amp",
    channel_numbers: int | list[int] = 1,
    t0: float = 0.0,
    t1: float = float("inf"),
) -> tuple[np.ndarray, int, float]:
    """Read data from an Intan RHD2000 directory (one file per channel format).

    Parameters
    ----------
    directoryname : str or Path
        Path to the RHD2000 data directory.
    header : dict or None
        Header information. If None, read from ``info.rhd`` in the directory.
    channel_type : str or int
        Type of channel to read:

        ==============================  =========
        Value                           Meaning
        ==============================  =========
        ``'time'``, ``'timestamp'``, 1  timestamps
        ``'amp'``, ``'amplifier'``, 2   amplifier
        ``'aux'``, ``'aux_in'``, 3      auxiliary input
        ``'supply'``, 4                 supply voltages
        ``'temp'``, 5                   temperature sensor
        ``'adc'``, 6                    board ADC
        ``'din'``, ``'digital_in'``, 7  digital input
        ``'dout'``, ``'digital_out'``, 8 digital output
        ==============================  =========

    channel_numbers : int or list of int
        Channel numbers to read (1-based).
    t0 : float
        Start time in seconds (0 is the beginning of the recording).
    t1 : float
        End time in seconds (``inf`` reads to end of file).

    Returns
    -------
    data : numpy.ndarray
        Column matrix where each column contains samples from one channel.
    total_samples : int
        Total number of amplifier samples in the file.
    total_time : float
        Total duration of the recording in seconds.
    """
    directoryname = Path(directoryname)

    if header is None:
        header = read_Intan_RHD2000_header(directoryname / "info.rhd")

    if isinstance(channel_numbers, int):
        channel_numbers = [channel_numbers]

    # Determine total samples from time.dat file size
    time_file = directoryname / "time.dat"
    if not time_file.exists():
        raise FileNotFoundError(f"No file {time_file}, required file.")

    total_samples = time_file.stat().st_size // 4  # int32 = 4 bytes
    sr = header["frequency_parameters"]["amplifier_sample_rate"]
    total_time = total_samples / sr

    # Fix t0, t1 to be in range
    if t0 < 0:
        t0 = 0.0
    if t1 > total_time - 1.0 / sr:
        t1 = total_time - 1.0 / sr

    # Compute starting and ending samples (1-based in MATLAB, 0-based here)
    s0 = int(t0 * sr)
    s1 = int(t1 * sr)
    num_samples = s1 - s0 + 1

    # Resolve channel_type string to integer
    _channel_type_map = {
        "time": 1,
        "timestamp": 1,
        "amp": 2,
        "amplifier": 2,
        "aux": 3,
        "aux_in": 3,
        "supply": 4,
        "temp": 5,
        "adc": 6,
        "din": 7,
        "digital_in": 7,
        "dout": 8,
        "digital_out": 8,
    }

    if isinstance(channel_type, str):
        ct = _channel_type_map.get(channel_type)
        if ct is None:
            raise ValueError(f"Unknown channel_type '{channel_type}'.")
        channel_type_int = ct
    else:
        channel_type_int = int(channel_type)

    # Header keys, file prefixes, and conversion parameters per channel type
    relevant_headers = {
        2: "amplifier_channels",
        3: "aux_input_channels",
        4: "supply_voltage_channels",
        5: None,  # temp - not yet supported in directory mode
        6: "board_adc_channels",
        7: "board_dig_in_channels",
        8: "board_dig_out_channels",
    }
    file_prefixes = {
        2: "amp-",
        3: "aux-",
        4: "vdd-",
        5: "temp",
        6: "board-",
        7: "board-",
        8: "board-",
    }
    sample_size_bytes = {1: 4, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
    sample_precision = {
        1: np.int32,
        2: np.int16,
        3: np.uint16,
        4: np.uint16,
        5: np.uint16,
        6: np.uint16,
        7: np.uint16,
    }
    conversion_scale = {1: 0, 2: 0.195, 3: 0.0000374, 4: 0.0000748, 5: 0, 6: 0.000050354, 7: 1, 8: 1}
    conversion_shift = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    eval_board_mode = header.get("eval_board_mode", 0)
    if eval_board_mode != 0:
        conversion_shift[6] = -32768
        conversion_scale[6] = 0.0003125

    data = np.empty((num_samples, 0), dtype=np.float64)

    if channel_type_int == 1:
        # Time channel
        if len(channel_numbers) != 1:
            raise ValueError(f"Only 1 time channel, {channel_numbers} requested.")
        with open(time_file, "rb") as fid:
            fid.seek(4 * s0)
            raw = np.fromfile(fid, dtype=np.dtype("<i4"), count=num_samples)
        data = (raw.astype(np.float64) / sr).reshape(-1, 1)

    elif channel_type_int in (2, 3, 4, 6, 7, 8):
        hinfo_key = relevant_headers[channel_type_int]
        if hinfo_key is None:
            raise NotImplementedError("Temperature reading not supported in directory mode.")
        hinfo = header[hinfo_key]
        prefix = file_prefixes[channel_type_int]
        sbytes = sample_size_bytes.get(channel_type_int, 2)
        dtype = np.dtype(sample_precision.get(channel_type_int, np.uint16)).newbyteorder("<")
        scale = conversion_scale[channel_type_int]
        shift = conversion_shift[channel_type_int]

        columns = []
        for ch_num in channel_numbers:
            if ch_num < 1 or ch_num > len(hinfo):
                raise IndexError(
                    f"Channel {ch_num} not in range 1 ... {len(hinfo)} listed in header."
                )
            ch_info = hinfo[ch_num - 1]  # convert 1-based to 0-based
            fname = f"{prefix}{ch_info['custom_channel_name']}.dat"
            fpath = directoryname / fname
            with open(fpath, "rb") as fid:
                fid.seek(sbytes * s0)
                raw = np.fromfile(fid, dtype=dtype, count=num_samples)
            raw_f = raw.astype(np.float64)
            if shift != 0:
                raw_f -= shift
            raw_f *= scale
            columns.append(raw_f)

        data = np.column_stack(columns) if columns else np.empty((num_samples, 0))

    elif channel_type_int == 5:
        raise NotImplementedError("Do not know how to read temperature in this mode yet.")

    return data, total_samples, total_time
