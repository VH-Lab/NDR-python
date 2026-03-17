"""Read an Intan RHD2000 data file and return all variables in a dict.

Port of +ndr/+format/+intan/+manufacturer/read_Intan_RHD2000_file_var.m

This is a modification of the manufacturer's code that accepts a filename
and returns the variables in a dict (analogous to MATLAB struct).
"""

from __future__ import annotations

import struct as _struct
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.intan.fread_QString import fread_QString


def read_Intan_RHD2000_file_var(filename: str | Path) -> dict[str, Any]:
    """Read an Intan RHD2000 data file and return all data as a dict.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.rhd`` data file.

    Returns
    -------
    dict
        Dictionary containing header information, channel metadata, and all
        recorded data arrays with appropriate scaling applied.
    """
    filename = Path(filename)
    filesize = filename.stat().st_size

    with open(filename, "rb") as fid:
        # Check magic number
        magic_number = _struct.unpack("<I", fid.read(4))[0]
        if magic_number != 0xC6912702:
            raise ValueError("Unrecognized file type.")

        # Version
        data_file_main_version_number = _struct.unpack("<h", fid.read(2))[0]
        data_file_secondary_version_number = _struct.unpack("<h", fid.read(2))[0]

        if data_file_main_version_number == 1:
            num_samples_per_data_block = 60
        else:
            num_samples_per_data_block = 128

        # Sampling rate and frequency settings
        sample_rate = _struct.unpack("<f", fid.read(4))[0]
        dsp_enabled = _struct.unpack("<h", fid.read(2))[0]
        actual_dsp_cutoff_frequency = _struct.unpack("<f", fid.read(4))[0]
        actual_lower_bandwidth = _struct.unpack("<f", fid.read(4))[0]
        actual_upper_bandwidth = _struct.unpack("<f", fid.read(4))[0]
        desired_dsp_cutoff_frequency = _struct.unpack("<f", fid.read(4))[0]
        desired_lower_bandwidth = _struct.unpack("<f", fid.read(4))[0]
        desired_upper_bandwidth = _struct.unpack("<f", fid.read(4))[0]

        # Notch filter
        notch_filter_mode = _struct.unpack("<h", fid.read(2))[0]
        notch_filter_frequency = 0
        if notch_filter_mode == 1:
            notch_filter_frequency = 50
        elif notch_filter_mode == 2:
            notch_filter_frequency = 60

        desired_impedance_test_frequency = _struct.unpack("<f", fid.read(4))[0]
        actual_impedance_test_frequency = _struct.unpack("<f", fid.read(4))[0]

        # Notes
        notes = {
            "note1": fread_QString(fid),
            "note2": fread_QString(fid),
            "note3": fread_QString(fid),
        }

        # Temperature sensor channels (v1.1+)
        num_temp_sensor_channels = 0
        if (
            (data_file_main_version_number == 1 and data_file_secondary_version_number >= 1)
            or data_file_main_version_number > 1
        ):
            num_temp_sensor_channels = _struct.unpack("<h", fid.read(2))[0]

        # Board mode (v1.3+)
        board_mode = 0
        if (
            (data_file_main_version_number == 1 and data_file_secondary_version_number >= 3)
            or data_file_main_version_number > 1
        ):
            board_mode = _struct.unpack("<h", fid.read(2))[0]

        # Reference channel (v2.0+)
        reference_channel = ""
        if data_file_main_version_number > 1:
            reference_channel = fread_QString(fid)

        # Frequency parameters
        frequency_parameters = {
            "amplifier_sample_rate": sample_rate,
            "aux_input_sample_rate": sample_rate / 4.0,
            "supply_voltage_sample_rate": sample_rate / num_samples_per_data_block,
            "board_adc_sample_rate": sample_rate,
            "board_dig_in_sample_rate": sample_rate,
            "desired_dsp_cutoff_frequency": desired_dsp_cutoff_frequency,
            "actual_dsp_cutoff_frequency": actual_dsp_cutoff_frequency,
            "dsp_enabled": dsp_enabled,
            "desired_lower_bandwidth": desired_lower_bandwidth,
            "actual_lower_bandwidth": actual_lower_bandwidth,
            "desired_upper_bandwidth": desired_upper_bandwidth,
            "actual_upper_bandwidth": actual_upper_bandwidth,
            "notch_filter_frequency": notch_filter_frequency,
            "desired_impedance_test_frequency": desired_impedance_test_frequency,
            "actual_impedance_test_frequency": actual_impedance_test_frequency,
        }

        # Read signal summary from header
        amplifier_channels: list[dict] = []
        aux_input_channels: list[dict] = []
        supply_voltage_channels: list[dict] = []
        board_adc_channels: list[dict] = []
        board_dig_in_channels: list[dict] = []
        board_dig_out_channels: list[dict] = []
        spike_triggers: list[dict] = []

        number_of_signal_groups = _struct.unpack("<h", fid.read(2))[0]
        for signal_group in range(1, number_of_signal_groups + 1):
            signal_group_name = fread_QString(fid)
            signal_group_prefix = fread_QString(fid)
            signal_group_enabled = _struct.unpack("<h", fid.read(2))[0]
            signal_group_num_channels = _struct.unpack("<h", fid.read(2))[0]
            _signal_group_num_amp_channels = _struct.unpack("<h", fid.read(2))[0]

            if signal_group_num_channels > 0 and signal_group_enabled > 0:
                for _signal_channel in range(signal_group_num_channels):
                    new_channel: dict[str, Any] = {
                        "port_name": signal_group_name,
                        "port_prefix": signal_group_prefix,
                        "port_number": signal_group,
                    }
                    new_channel["native_channel_name"] = fread_QString(fid)
                    new_channel["custom_channel_name"] = fread_QString(fid)
                    new_channel["native_order"] = _struct.unpack("<h", fid.read(2))[0]
                    new_channel["custom_order"] = _struct.unpack("<h", fid.read(2))[0]
                    signal_type = _struct.unpack("<h", fid.read(2))[0]
                    channel_enabled = _struct.unpack("<h", fid.read(2))[0]
                    new_channel["chip_channel"] = _struct.unpack("<h", fid.read(2))[0]
                    new_channel["board_stream"] = _struct.unpack("<h", fid.read(2))[0]
                    new_trigger = {
                        "voltage_trigger_mode": _struct.unpack("<h", fid.read(2))[0],
                        "voltage_threshold": _struct.unpack("<h", fid.read(2))[0],
                        "digital_trigger_channel": _struct.unpack("<h", fid.read(2))[0],
                        "digital_edge_polarity": _struct.unpack("<h", fid.read(2))[0],
                    }
                    new_channel["electrode_impedance_magnitude"] = _struct.unpack("<f", fid.read(4))[0]
                    new_channel["electrode_impedance_phase"] = _struct.unpack("<f", fid.read(4))[0]

                    if channel_enabled:
                        if signal_type == 0:
                            amplifier_channels.append(new_channel)
                            spike_triggers.append(new_trigger)
                        elif signal_type == 1:
                            aux_input_channels.append(new_channel)
                        elif signal_type == 2:
                            supply_voltage_channels.append(new_channel)
                        elif signal_type == 3:
                            board_adc_channels.append(new_channel)
                        elif signal_type == 4:
                            board_dig_in_channels.append(new_channel)
                        elif signal_type == 5:
                            board_dig_out_channels.append(new_channel)
                        else:
                            raise ValueError("Unknown channel type")

        num_amplifier_channels = len(amplifier_channels)
        num_aux_input_channels = len(aux_input_channels)
        num_supply_voltage_channels = len(supply_voltage_channels)
        num_board_adc_channels = len(board_adc_channels)
        num_board_dig_in_channels = len(board_dig_in_channels)
        num_board_dig_out_channels = len(board_dig_out_channels)

        # Calculate bytes per data block
        bytes_per_block = num_samples_per_data_block * 4  # timestamps
        bytes_per_block += num_samples_per_data_block * 2 * num_amplifier_channels
        bytes_per_block += (num_samples_per_data_block // 4) * 2 * num_aux_input_channels
        bytes_per_block += 1 * 2 * num_supply_voltage_channels
        bytes_per_block += num_samples_per_data_block * 2 * num_board_adc_channels
        if num_board_dig_in_channels > 0:
            bytes_per_block += num_samples_per_data_block * 2
        if num_board_dig_out_channels > 0:
            bytes_per_block += num_samples_per_data_block * 2
        if num_temp_sensor_channels > 0:
            bytes_per_block += 1 * 2 * num_temp_sensor_channels

        # How many data blocks remain?
        bytes_remaining = filesize - fid.tell()
        data_present = bytes_remaining > 0
        num_data_blocks = bytes_remaining // bytes_per_block

        num_amplifier_samples = num_samples_per_data_block * num_data_blocks
        num_aux_input_samples = (num_samples_per_data_block // 4) * num_data_blocks
        num_supply_voltage_samples = num_data_blocks
        num_board_adc_samples = num_samples_per_data_block * num_data_blocks
        num_board_dig_in_samples = num_samples_per_data_block * num_data_blocks
        num_board_dig_out_samples = num_samples_per_data_block * num_data_blocks

        # Pre-allocate and read data
        t_amplifier = np.zeros(num_amplifier_samples, dtype=np.float64)
        amplifier_data = np.zeros((num_amplifier_channels, num_amplifier_samples), dtype=np.float64)
        aux_input_data = np.zeros((num_aux_input_channels, num_aux_input_samples), dtype=np.float64)
        supply_voltage_data = np.zeros(
            (num_supply_voltage_channels, num_supply_voltage_samples), dtype=np.float64
        )
        temp_sensor_data = np.zeros(
            (num_temp_sensor_channels, num_supply_voltage_samples), dtype=np.float64
        )
        board_adc_data = np.zeros((num_board_adc_channels, num_board_adc_samples), dtype=np.float64)
        board_dig_in_data = np.zeros(
            (num_board_dig_in_channels, num_board_dig_in_samples), dtype=np.float64
        )
        board_dig_in_raw = np.zeros(num_board_dig_in_samples, dtype=np.uint16)
        board_dig_out_data = np.zeros(
            (num_board_dig_out_channels, num_board_dig_out_samples), dtype=np.float64
        )
        board_dig_out_raw = np.zeros(num_board_dig_out_samples, dtype=np.uint16)

        if data_present:
            # Determine timestamp dtype based on version
            if (
                (data_file_main_version_number == 1 and data_file_secondary_version_number >= 2)
                or data_file_main_version_number > 1
            ):
                ts_dtype = np.dtype("<i4")  # int32
            else:
                ts_dtype = np.dtype("<u4")  # uint32

            amp_idx = 0
            aux_idx = 0
            sv_idx = 0
            adc_idx = 0
            din_idx = 0
            dout_idx = 0
            spb = num_samples_per_data_block
            aux_spb = spb // 4

            for _block in range(num_data_blocks):
                # Timestamps
                t_amplifier[amp_idx : amp_idx + spb] = np.frombuffer(
                    fid.read(4 * spb), dtype=ts_dtype
                )

                # Amplifier data
                if num_amplifier_channels > 0:
                    raw = np.frombuffer(
                        fid.read(2 * num_amplifier_channels * spb), dtype=np.dtype("<u2")
                    ).reshape(spb, num_amplifier_channels)
                    amplifier_data[:, amp_idx : amp_idx + spb] = raw.T.astype(np.float64)

                # Aux input data
                if num_aux_input_channels > 0:
                    raw = np.frombuffer(
                        fid.read(2 * num_aux_input_channels * aux_spb), dtype=np.dtype("<u2")
                    ).reshape(aux_spb, num_aux_input_channels)
                    aux_input_data[:, aux_idx : aux_idx + aux_spb] = raw.T.astype(np.float64)

                # Supply voltage data
                if num_supply_voltage_channels > 0:
                    raw = np.frombuffer(
                        fid.read(2 * num_supply_voltage_channels), dtype=np.dtype("<u2")
                    )
                    supply_voltage_data[:, sv_idx] = raw.astype(np.float64)

                # Temperature sensor data
                if num_temp_sensor_channels > 0:
                    raw = np.frombuffer(
                        fid.read(2 * num_temp_sensor_channels), dtype=np.dtype("<i2")
                    )
                    temp_sensor_data[:, sv_idx] = raw.astype(np.float64)

                # Board ADC data
                if num_board_adc_channels > 0:
                    raw = np.frombuffer(
                        fid.read(2 * num_board_adc_channels * spb), dtype=np.dtype("<u2")
                    ).reshape(spb, num_board_adc_channels)
                    board_adc_data[:, adc_idx : adc_idx + spb] = raw.T.astype(np.float64)

                # Board digital in
                if num_board_dig_in_channels > 0:
                    board_dig_in_raw[din_idx : din_idx + spb] = np.frombuffer(
                        fid.read(2 * spb), dtype=np.dtype("<u2")
                    )

                # Board digital out
                if num_board_dig_out_channels > 0:
                    board_dig_out_raw[dout_idx : dout_idx + spb] = np.frombuffer(
                        fid.read(2 * spb), dtype=np.dtype("<u2")
                    )

                amp_idx += spb
                aux_idx += aux_spb
                sv_idx += 1
                adc_idx += spb
                din_idx += spb
                dout_idx += spb

    # Post-processing (outside file context)
    if data_present:
        # Extract digital channels
        for i, ch in enumerate(board_dig_in_channels):
            mask = 1 << ch["native_order"]
            board_dig_in_data[i, :] = ((board_dig_in_raw & mask) > 0).astype(np.float64)

        for i, ch in enumerate(board_dig_out_channels):
            mask = 1 << ch["native_order"]
            board_dig_out_data[i, :] = ((board_dig_out_raw & mask) > 0).astype(np.float64)

        # Scale voltage levels
        amplifier_data = 0.195 * (amplifier_data - 32768.0)  # microvolts
        aux_input_data = 37.4e-6 * aux_input_data  # volts
        supply_voltage_data = 74.8e-6 * supply_voltage_data  # volts

        if board_mode == 1:
            board_adc_data = 152.59e-6 * (board_adc_data - 32768.0)  # volts
        elif board_mode == 13:  # Intan Recording Controller
            board_adc_data = 312.5e-6 * (board_adc_data - 32768.0)  # volts
        else:
            board_adc_data = 50.354e-6 * board_adc_data  # volts

        temp_sensor_data = temp_sensor_data / 100.0  # degrees C

        # Scale time
        t_amplifier = t_amplifier / sample_rate
        t_aux_input = t_amplifier[::4]
        t_supply_voltage = t_amplifier[::num_samples_per_data_block]
        t_board_adc = t_amplifier.copy()
        t_dig = t_amplifier.copy()
        t_temp_sensor = t_supply_voltage.copy()

    # Build output dict
    result: dict[str, Any] = {
        "filename": str(filename),
        "notes": notes,
        "frequency_parameters": frequency_parameters,
    }

    if data_file_main_version_number > 1:
        result["reference_channel"] = reference_channel

    if num_amplifier_channels > 0:
        result["amplifier_channels"] = amplifier_channels
        if data_present:
            result["amplifier_data"] = amplifier_data
            result["t_amplifier"] = t_amplifier
        result["spike_triggers"] = spike_triggers

    if num_aux_input_channels > 0:
        result["aux_input_channels"] = aux_input_channels
        if data_present:
            result["aux_input_data"] = aux_input_data
            result["t_aux_input"] = t_aux_input

    if num_supply_voltage_channels > 0:
        result["supply_voltage_channels"] = supply_voltage_channels
        if data_present:
            result["supply_voltage_data"] = supply_voltage_data
            result["t_supply_voltage"] = t_supply_voltage

    if num_board_adc_channels > 0:
        result["board_adc_channels"] = board_adc_channels
        if data_present:
            result["board_adc_data"] = board_adc_data
            result["t_board_adc"] = t_board_adc

    if num_board_dig_in_channels > 0:
        result["board_dig_in_channels"] = board_dig_in_channels
        if data_present:
            result["board_dig_in_data"] = board_dig_in_data
            result["t_dig"] = t_dig

    if num_board_dig_out_channels > 0:
        result["board_dig_out_channels"] = board_dig_out_channels
        if data_present:
            result["board_dig_out_data"] = board_dig_out_data
            result["t_dig"] = t_dig

    if num_temp_sensor_channels > 0:
        if data_present:
            result["temp_sensor_data"] = temp_sensor_data
            result["t_temp_sensor"] = t_temp_sensor

    return result
