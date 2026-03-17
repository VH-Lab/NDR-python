"""Read Intan RHD2000 file headers.

Port of +ndr/+format/+intan/read_Intan_RHD2000_header.m

This module uses vhlab-toolbox-python's Intan reading capability when available,
and falls back to a native implementation otherwise.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def _fread_QString(f) -> str:
    """Read a Qt QString from a binary file (port of fread_QString.m)."""
    length_bytes = f.read(4)
    if len(length_bytes) < 4:
        return ""
    length = struct.unpack("<I", length_bytes)[0]
    if length == 0xFFFFFFFF:
        return ""
    data = f.read(length)
    return data.decode("utf-16-le", errors="replace")


def _read_channel_header(f, data_file_main_version: int) -> dict[str, Any]:
    """Read a single channel header entry."""
    ch = {}
    ch["native_channel_name"] = _fread_QString(f)
    ch["custom_channel_name"] = _fread_QString(f)
    ch["native_order"] = struct.unpack("<i", f.read(4))[0]
    ch["custom_order"] = struct.unpack("<i", f.read(4))[0]
    ch["signal_type"] = struct.unpack("<i", f.read(4))[0]
    ch["channel_enabled"] = struct.unpack("<h", f.read(2))[0]
    ch["chip_channel"] = struct.unpack("<h", f.read(2))[0]
    if data_file_main_version > 0:
        ch["command_stream"] = struct.unpack("<h", f.read(2))[0]
    ch["board_stream"] = struct.unpack("<h", f.read(2))[0]
    ch["spike_scope_trigger_mode"] = struct.unpack("<h", f.read(2))[0]
    ch["spike_scope_voltage_thresh"] = struct.unpack("<h", f.read(2))[0]
    ch["spike_scope_digital_trigger_channel"] = struct.unpack("<h", f.read(2))[0]
    ch["spike_scope_digital_edge_polarity"] = struct.unpack("<h", f.read(2))[0]
    ch["electrode_impedance_magnitude"] = struct.unpack("<f", f.read(4))[0]
    ch["electrode_impedance_phase"] = struct.unpack("<f", f.read(4))[0]
    return ch


def read_Intan_RHD2000_header(filename: str | Path) -> dict[str, Any]:
    """Read the header from an Intan RHD2000 file.

    Parameters
    ----------
    filename : str or Path
        Path to the .rhd file.

    Returns
    -------
    dict
        Header information including frequency parameters and channel lists.
    """
    filename = Path(filename)
    header: dict[str, Any] = {}

    with open(filename, "rb") as f:
        # Magic number
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0xC6912702:
            raise ValueError(f"Not a valid Intan RHD2000 file: bad magic number {magic:#x}")

        # Version
        header["data_file_main_version_number"] = struct.unpack("<h", f.read(2))[0]
        header["data_file_secondary_version_number"] = struct.unpack("<h", f.read(2))[0]

        main_version = header["data_file_main_version_number"]

        # Frequency parameters
        freq = {}
        freq["amplifier_sample_rate"] = struct.unpack("<f", f.read(4))[0]
        freq["aux_input_sample_rate"] = freq["amplifier_sample_rate"] / 4.0
        freq["supply_voltage_sample_rate"] = freq["amplifier_sample_rate"] / 60.0
        freq["board_adc_sample_rate"] = freq["amplifier_sample_rate"]
        freq["board_dig_in_sample_rate"] = freq["amplifier_sample_rate"]

        freq["dsp_enabled"] = struct.unpack("<h", f.read(2))[0]
        freq["actual_dsp_cutoff_frequency"] = struct.unpack("<f", f.read(4))[0]
        freq["actual_lower_bandwidth"] = struct.unpack("<f", f.read(4))[0]

        if main_version > 0:
            freq["actual_lower_settle_bandwidth"] = struct.unpack("<f", f.read(4))[0]

        freq["actual_upper_bandwidth"] = struct.unpack("<f", f.read(4))[0]
        freq["desired_dsp_cutoff_frequency"] = struct.unpack("<f", f.read(4))[0]
        freq["desired_lower_bandwidth"] = struct.unpack("<f", f.read(4))[0]

        if main_version > 0:
            freq["desired_lower_settle_bandwidth"] = struct.unpack("<f", f.read(4))[0]

        freq["desired_upper_bandwidth"] = struct.unpack("<f", f.read(4))[0]

        # Notch filter mode
        freq["notch_filter_mode"] = struct.unpack("<h", f.read(2))[0]
        freq["desired_impedance_test_frequency"] = struct.unpack("<f", f.read(4))[0]
        freq["actual_impedance_test_frequency"] = struct.unpack("<f", f.read(4))[0]

        if main_version > 1:
            freq["amp_settle_mode"] = struct.unpack("<h", f.read(2))[0]
            freq["charge_recovery_mode"] = struct.unpack("<h", f.read(2))[0]

        header["frequency_parameters"] = freq

        # Note
        if main_version > 0:
            header["note1"] = _fread_QString(f)
            header["note2"] = _fread_QString(f)
            header["note3"] = _fread_QString(f)

        # DC amplifier data saved
        if main_version > 1:
            header["dc_amplifier_data_saved"] = struct.unpack("<h", f.read(2))[0]
        else:
            header["dc_amplifier_data_saved"] = 0

        # Eval board mode
        if main_version > 1:
            header["eval_board_mode"] = struct.unpack("<h", f.read(2))[0]
        else:
            header["eval_board_mode"] = 0

        # Reference channel (v2.0+)
        if main_version > 0:
            header["reference_channel"] = _fread_QString(f)

        # Number of signal groups
        num_signal_groups = struct.unpack("<h", f.read(2))[0]

        # Initialize channel lists
        header["amplifier_channels"] = []
        header["aux_input_channels"] = []
        header["supply_voltage_channels"] = []
        header["board_adc_channels"] = []
        header["board_dig_in_channels"] = []
        header["board_dig_out_channels"] = []

        for _g in range(num_signal_groups):
            _group_name = _fread_QString(f)
            _group_prefix = _fread_QString(f)
            _group_enabled = struct.unpack("<h", f.read(2))[0]
            num_channels_in_group = struct.unpack("<h", f.read(2))[0]
            _num_amplifier_channels_in_group = struct.unpack("<h", f.read(2))[0]

            for _c in range(num_channels_in_group):
                ch = _read_channel_header(f, main_version)

                signal_type = ch["signal_type"]
                if ch["channel_enabled"]:
                    if signal_type == 0:
                        header["amplifier_channels"].append(ch)
                    elif signal_type == 1:
                        header["aux_input_channels"].append(ch)
                    elif signal_type == 2:
                        header["supply_voltage_channels"].append(ch)
                    elif signal_type == 3:
                        header["board_adc_channels"].append(ch)
                    elif signal_type == 4:
                        header["board_dig_in_channels"].append(ch)
                    elif signal_type == 5:
                        header["board_dig_out_channels"].append(ch)

    return header
