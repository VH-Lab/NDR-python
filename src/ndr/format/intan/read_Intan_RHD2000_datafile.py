"""Read data from Intan RHD2000 single-file format.

Port of +ndr/+format/+intan/read_Intan_RHD2000_datafile.m
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header


def Intan_RHD2000_blockinfo(
    filename: str | Path, header: dict[str, Any]
) -> tuple[dict[str, Any], int, int, int]:
    """Get block structure information for an RHD file.

    Parameters
    ----------
    filename : str or Path
        Path to the .rhd file.
    header : dict
        Header from read_Intan_RHD2000_header.

    Returns
    -------
    tuple of (blockinfo, bytes_per_block, bytes_present, num_data_blocks)
    """
    filename = Path(filename)
    filesize = filename.stat().st_size

    num_amplifier = len(header.get("amplifier_channels", []))
    num_aux = len(header.get("aux_input_channels", []))
    num_supply = len(header.get("supply_voltage_channels", []))
    num_adc = len(header.get("board_adc_channels", []))
    num_dig_in = len(header.get("board_dig_in_channels", []))
    num_dig_out = len(header.get("board_dig_out_channels", []))
    dc_amp_saved = header.get("dc_amplifier_data_saved", 0)

    samples_per_block = 60  # Intan RHD uses 60 samples per data block

    # Calculate bytes per data block
    # timestamp: 4 bytes * 60 samples
    bytes_per_block = 4 * samples_per_block

    # amplifier data: 2 bytes * num_channels * 60 samples
    bytes_per_block += 2 * num_amplifier * samples_per_block

    # DC amplifier data (if saved): 2 bytes * num_channels * 60 samples
    if dc_amp_saved:
        bytes_per_block += 2 * num_amplifier * samples_per_block

    # Stim data (v3+): skip for now, most common is v1/v2

    # aux data: 2 bytes * num_aux * 15 samples (sampled at 1/4 rate)
    bytes_per_block += 2 * num_aux * (samples_per_block // 4)

    # supply voltage: 2 bytes * num_supply * 1 sample (sampled at 1/60 rate)
    bytes_per_block += 2 * num_supply

    # temp sensor: 2 bytes * num_supply * 1 sample
    bytes_per_block += 2 * num_supply

    # board ADC: 2 bytes * num_adc * 60 samples
    bytes_per_block += 2 * num_adc * samples_per_block

    # board digital in: 2 bytes * 60 samples (one uint16 per sample for all channels)
    if num_dig_in > 0:
        bytes_per_block += 2 * samples_per_block

    # board digital out: 2 bytes * 60 samples
    if num_dig_out > 0:
        bytes_per_block += 2 * samples_per_block

    # Calculate header size (approximate - read until data starts)
    # We need to find where the header ends
    header_size = _get_header_size(filename, header)

    bytes_present = filesize - header_size
    num_data_blocks = bytes_present // bytes_per_block

    blockinfo = {
        "samples_per_block": samples_per_block,
        "num_amplifier": num_amplifier,
        "num_aux": num_aux,
        "num_supply": num_supply,
        "num_adc": num_adc,
        "num_dig_in": num_dig_in,
        "num_dig_out": num_dig_out,
        "dc_amp_saved": dc_amp_saved,
        "header_size": header_size,
    }

    return blockinfo, bytes_per_block, bytes_present, num_data_blocks


def _get_header_size(filename: Path, header: dict[str, Any]) -> int:
    """Determine the header size by reading and counting bytes."""
    # Re-read the file and track position after header parsing
    with open(filename, "rb") as f:
        # Magic number + version
        f.read(4 + 2 + 2)

        main_version = header["data_file_main_version_number"]
        secondary_version = header["data_file_secondary_version_number"]

        # Sample rate
        f.read(4)

        # DSP fields: dsp_enabled(2) + actual_dsp_cutoff(4) + actual_lower_bw(4)
        # + actual_upper_bw(4) + desired_dsp_cutoff(4) + desired_lower_bw(4)
        # + desired_upper_bw(4)
        f.read(2 + 4 + 4 + 4 + 4 + 4 + 4)

        # Notch filter mode(2) + desired_impedance_test_freq(4) + actual_impedance_test_freq(4)
        f.read(2 + 4 + 4)

        # Notes (always present)
        for _ in range(3):
            _read_qstring_skip(f)

        # Temperature sensor channels (v1.1+)
        if (main_version == 1 and secondary_version >= 1) or main_version > 1:
            f.read(2)

        # Eval board mode (v1.3+)
        if (main_version == 1 and secondary_version >= 3) or main_version > 1:
            f.read(2)

        # Reference channel (v2.0+)
        if main_version > 1:
            _read_qstring_skip(f)

        # Signal groups
        num_signal_groups = struct.unpack("<h", f.read(2))[0]

        for _ in range(num_signal_groups):
            _read_qstring_skip(f)  # group name
            _read_qstring_skip(f)  # group prefix
            f.read(2)  # enabled
            num_channels = struct.unpack("<h", f.read(2))[0]
            f.read(2)  # num_amplifier_channels

            for _ in range(num_channels):
                _read_qstring_skip(f)  # native_channel_name
                _read_qstring_skip(f)  # custom_channel_name
                # native_order(2) + custom_order(2) + signal_type(2)
                # + enabled(2) + chip_channel(2) + board_stream(2)
                # + voltage_trigger_mode(2) + voltage_threshold(2)
                # + digital_trigger_channel(2) + digital_edge_polarity(2)
                # + impedance_magnitude(4) + impedance_phase(4)
                f.read(2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 4 + 4)

        return f.tell()


def _read_qstring_skip(f) -> None:
    """Skip a Qt QString in a binary file."""
    length_bytes = f.read(4)
    if len(length_bytes) < 4:
        return
    length = struct.unpack("<I", length_bytes)[0]
    if length == 0xFFFFFFFF:
        return
    f.read(length)


def read_Intan_RHD2000_datafile(
    filename: str | Path,
    header_filename: str = "",
    channeltype: str = "amp",
    channel: int | list[int] = 1,
    t0: float = 0.0,
    t1: float = float("inf"),
) -> np.ndarray:
    """Read data from an Intan RHD2000 single-file format.

    Parameters
    ----------
    filename : str or Path
        Path to the .rhd file.
    header_filename : str
        Unused (for API compatibility).
    channeltype : str
        Channel type: 'amp', 'aux', 'din', 'dout', 'adc', 'time'.
    channel : int or list of int
        Channel number(s) (1-based).
    t0 : float
        Start time in seconds.
    t1 : float
        End time in seconds.

    Returns
    -------
    numpy.ndarray
        Data array with one column per channel.
    """
    filename = Path(filename)
    header = read_Intan_RHD2000_header(filename)
    blockinfo, bytes_per_block, bytes_present, num_data_blocks = Intan_RHD2000_blockinfo(
        filename, header
    )

    if isinstance(channel, int):
        channel = [channel]

    sr = header["frequency_parameters"]["amplifier_sample_rate"]
    samples_per_block = blockinfo["samples_per_block"]
    total_samples = samples_per_block * num_data_blocks

    if channeltype == "aux":
        sr_actual = header["frequency_parameters"]["aux_input_sample_rate"]
    else:
        sr_actual = sr

    # Convert time to sample indices
    s0 = max(0, int(round(t0 * sr_actual)))
    if t1 == float("inf"):
        if channeltype == "aux":
            s1 = (samples_per_block // 4) * num_data_blocks - 1
        else:
            s1 = total_samples - 1
    else:
        s1 = int(round(t1 * sr_actual))

    num_samples = s1 - s0 + 1
    if num_samples <= 0:
        return np.array([]).reshape(0, len(channel))

    # Read the entire data section and extract requested channels
    num_amplifier = blockinfo["num_amplifier"]
    num_aux = blockinfo["num_aux"]
    num_supply = blockinfo["num_supply"]
    num_adc = blockinfo["num_adc"]
    num_dig_in = blockinfo["num_dig_in"]
    num_dig_out = blockinfo["num_dig_out"]
    dc_amp_saved = blockinfo["dc_amp_saved"]
    header_size = blockinfo["header_size"]

    with open(filename, "rb") as f:
        f.seek(header_size)

        # Pre-allocate output arrays
        if channeltype == "time":
            all_data = np.zeros(total_samples, dtype=np.float64)
        elif channeltype == "amp":
            all_data = np.zeros((total_samples, num_amplifier), dtype=np.float64)
        elif channeltype == "aux":
            aux_samples = (samples_per_block // 4) * num_data_blocks
            all_data = np.zeros((aux_samples, num_aux), dtype=np.float64)
        elif channeltype in ("din", "dout"):
            all_data = np.zeros((total_samples, 16), dtype=np.float64)
        elif channeltype == "adc":
            all_data = np.zeros((total_samples, num_adc), dtype=np.float64)
        else:
            raise ValueError(f"Unknown channel type '{channeltype}'.")

        for block in range(num_data_blocks):
            block_start = block * samples_per_block

            # Timestamps: int32 x 60
            ts_raw = np.frombuffer(f.read(4 * samples_per_block), dtype=np.int32)

            if channeltype == "time":
                all_data[block_start : block_start + samples_per_block] = ts_raw / sr

            # Amplifier data: int16 x num_amplifier x 60
            if num_amplifier > 0:
                amp_raw = (
                    np.frombuffer(f.read(2 * num_amplifier * samples_per_block), dtype=np.uint16)
                    .reshape(num_amplifier, samples_per_block)
                    .T
                )

                if channeltype == "amp":
                    # Convert to microvolts: (raw - 32768) * 0.195
                    all_data[block_start : block_start + samples_per_block, :] = (
                        amp_raw.astype(np.float64) - 32768.0
                    ) * 0.195

            # DC amplifier data
            if dc_amp_saved and num_amplifier > 0:
                f.read(2 * num_amplifier * samples_per_block)

            # Stim data (for RHS files, not RHD - skip)

            # Aux input data: uint16 x num_aux x 15
            if num_aux > 0:
                aux_per_block = samples_per_block // 4
                aux_raw = (
                    np.frombuffer(f.read(2 * num_aux * aux_per_block), dtype=np.uint16)
                    .reshape(num_aux, aux_per_block)
                    .T
                )

                if channeltype == "aux":
                    aux_block_start = block * aux_per_block
                    all_data[aux_block_start : aux_block_start + aux_per_block, :] = (
                        aux_raw.astype(np.float64) * 37.4e-6
                    )

            # Supply voltage: uint16 x num_supply x 1
            if num_supply > 0:
                f.read(2 * num_supply)

            # Temp sensor
            if num_supply > 0:
                f.read(2 * num_supply)

            # Board ADC: uint16 x num_adc x 60
            if num_adc > 0:
                adc_raw = (
                    np.frombuffer(f.read(2 * num_adc * samples_per_block), dtype=np.uint16)
                    .reshape(num_adc, samples_per_block)
                    .T
                )

                if channeltype == "adc":
                    all_data[block_start : block_start + samples_per_block, :] = adc_raw.astype(
                        np.float64
                    )

            # Board digital inputs: uint16 x 60
            if num_dig_in > 0:
                dig_in_raw = np.frombuffer(f.read(2 * samples_per_block), dtype=np.uint16)
                if channeltype == "din":
                    for bit in range(16):
                        all_data[block_start : block_start + samples_per_block, bit] = (
                            (dig_in_raw >> bit) & 1
                        ).astype(np.float64)

            # Board digital outputs: uint16 x 60
            if num_dig_out > 0:
                dig_out_raw = np.frombuffer(f.read(2 * samples_per_block), dtype=np.uint16)
                if channeltype == "dout":
                    for bit in range(16):
                        all_data[block_start : block_start + samples_per_block, bit] = (
                            (dig_out_raw >> bit) & 1
                        ).astype(np.float64)

    # Extract the requested time range and channels
    if channeltype == "time":
        return all_data[s0 : s1 + 1].reshape(-1, 1)
    else:
        # Convert 1-based channel numbers to 0-based indices
        ch_indices = [c - 1 for c in channel]
        return all_data[s0 : s1 + 1, :][:, ch_indices]
