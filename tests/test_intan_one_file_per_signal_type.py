"""Intan "one file per signal type" layout support.

MATLAB counterpart:
``tools/tests/+ndr/+unittest/+format/+intan/TestOneFilePerSignalType.m``.

Drives ``read_IntanRHD2000_one_file_per_channel_type`` and
``read_Intan_RHD2000_directory`` with synthetic ``.dat`` files: interleaved
multi-channel amplifier reads, an ``auxiliary.dat`` regression pin, digital
in/out returning the packed 16-bit word, sub-range slicing, prefix
resolution (Intan timestamp or lab prefix like ``febc0_u000_000_``), and
an end-to-end directory-reader path with a synthetic header covering the
per-signal-type dispatch and ``native_order``-driven digital bit extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ndr.format.intan.read_Intan_RHD2000_directory import read_Intan_RHD2000_directory
from ndr.format.intan.read_IntanRHD2000_one_file_per_channel_type import (
    read_IntanRHD2000_one_file_per_channel_type,
)
from ndr.reader.intan_rhd import ndr_reader_intan__rhd


def _write_dat(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    values.astype(dtype, copy=False).tofile(path)


# ---------------------------------------------------------------------------
# read_IntanRHD2000_one_file_per_channel_type
# ---------------------------------------------------------------------------


def test_read_amplifier_interleaved(tmp_path: Path) -> None:
    num_channels = 3
    num_samples = 5
    ch1 = np.array([100, 200, 300, 400, 500], dtype=np.int16)
    ch2 = np.array([-100, -200, -300, -400, -500], dtype=np.int16)
    ch3 = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    interleaved = np.empty(num_channels * num_samples, dtype=np.int16)
    interleaved[0::num_channels] = ch1
    interleaved[1::num_channels] = ch2
    interleaved[2::num_channels] = ch3
    _write_dat(tmp_path / "amplifier.dat", interleaved, np.dtype("<i2"))

    got1 = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 2, 1, num_channels, 1, num_samples)
    got2 = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 2, 2, num_channels, 1, num_samples)
    got3 = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 2, 3, num_channels, 1, num_samples)

    np.testing.assert_array_equal(got1, ch1)
    np.testing.assert_array_equal(got2, ch2)
    np.testing.assert_array_equal(got3, ch3)


def test_auxiliary_filename(tmp_path: Path) -> None:
    # Regression: MATLAB used to look for 'auxin.dat'; Intan writes 'auxiliary.dat'.
    num_channels = 2
    num_samples = 4
    a = np.array([10, 20, 30, 40], dtype=np.uint16)
    b = np.array([1000, 2000, 3000, 4000], dtype=np.uint16)
    interleaved = np.empty(num_channels * num_samples, dtype=np.uint16)
    interleaved[0::2] = a
    interleaved[1::2] = b
    _write_dat(tmp_path / "auxiliary.dat", interleaved, np.dtype("<u2"))

    got1 = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 3, 1, num_channels, 1, num_samples)
    got2 = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 3, 2, num_channels, 1, num_samples)

    np.testing.assert_array_equal(got1, a)
    np.testing.assert_array_equal(got2, b)


def test_time_channel(tmp_path: Path) -> None:
    times = np.arange(10, dtype=np.int32)
    _write_dat(tmp_path / "time.dat", times, np.dtype("<i4"))

    got = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 1, 1, 1, 1, times.size)
    np.testing.assert_array_equal(got, times)


def test_digital_returns_packed_word(tmp_path: Path) -> None:
    # Digital in/out is a single packed 16-bit word per sample: the helper
    # returns it verbatim; the caller extracts the requested bit.
    words = np.array([0, 1, 2, 4, 8, 16, 32, 0xFFFF], dtype=np.uint16)

    _write_dat(tmp_path / "digitalin.dat", words, np.dtype("<u2"))
    got = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 7, 1, 1, 1, words.size)
    np.testing.assert_array_equal(got, words)

    _write_dat(tmp_path / "digitalout.dat", words, np.dtype("<u2"))
    got = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 8, 1, 1, 1, words.size)
    np.testing.assert_array_equal(got, words)


def test_sub_range_slicing(tmp_path: Path) -> None:
    num_channels = 2
    num_samples = 8
    ch1 = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int16)
    ch2 = np.array([11, 22, 33, 44, 55, 66, 77, 88], dtype=np.int16)
    interleaved = np.empty(num_channels * num_samples, dtype=np.int16)
    interleaved[0::2] = ch1
    interleaved[1::2] = ch2
    _write_dat(tmp_path / "amplifier.dat", interleaved, np.dtype("<i2"))

    got = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 2, 2, num_channels, 3, 6)
    np.testing.assert_array_equal(got, ch2[2:6])


def test_prefixed_filename(tmp_path: Path) -> None:
    num_channels = 2
    num_samples = 3
    ch1 = np.array([7, 8, 9], dtype=np.int16)
    ch2 = np.array([70, 80, 90], dtype=np.int16)
    interleaved = np.empty(num_channels * num_samples, dtype=np.int16)
    interleaved[0::2] = ch1
    interleaved[1::2] = ch2
    _write_dat(tmp_path / "febc0_u000_000_amplifier.dat", interleaved, np.dtype("<i2"))

    got1 = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 2, 1, num_channels, 1, num_samples)
    got2 = read_IntanRHD2000_one_file_per_channel_type(tmp_path, 2, 2, num_channels, 1, num_samples)

    np.testing.assert_array_equal(got1, ch1)
    np.testing.assert_array_equal(got2, ch2)


def test_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_IntanRHD2000_one_file_per_channel_type(tmp_path, 2, 1, 1, 1, 1)


# ---------------------------------------------------------------------------
# read_Intan_RHD2000_directory end-to-end (with a synthetic header)
# ---------------------------------------------------------------------------


def _synthetic_header(
    sample_rate: float,
    amp_channels: list[dict],
    din_channels: list[dict],
) -> dict:
    return {
        "fileinfo": {"filename": "synthetic", "filesize": 0, "headersize": 0},
        "frequency_parameters": {
            "amplifier_sample_rate": sample_rate,
            "aux_input_sample_rate": sample_rate / 4,
            "supply_voltage_sample_rate": sample_rate / 128,
            "board_adc_sample_rate": sample_rate,
            "board_dig_in_sample_rate": sample_rate,
        },
        "amplifier_channels": amp_channels,
        "aux_input_channels": [],
        "supply_voltage_channels": [],
        "board_adc_channels": [],
        "board_dig_in_channels": din_channels,
        "board_dig_out_channels": [],
        "num_temp_sensor_channels": 0,
        "eval_board_mode": 0,
    }


def test_directory_reader_dispatches_to_per_signal_type(tmp_path: Path) -> None:
    sample_rate = 20000.0
    num_samples = 4
    times = np.arange(num_samples, dtype=np.int32)
    ch1 = np.array([100, 200, 300, 400], dtype=np.int16)
    ch2 = np.array([-100, -200, -300, -400], dtype=np.int16)
    amp = np.empty(2 * num_samples, dtype=np.int16)
    amp[0::2] = ch1
    amp[1::2] = ch2
    # bit 0 always on, bit 3 on for the last two samples.
    din_word = np.array([1, 1, 1 + 8, 1 + 8], dtype=np.uint16)

    _write_dat(tmp_path / "time.dat", times, np.dtype("<i4"))
    _write_dat(tmp_path / "amplifier.dat", amp, np.dtype("<i2"))
    _write_dat(tmp_path / "digitalin.dat", din_word, np.dtype("<u2"))

    header = _synthetic_header(
        sample_rate,
        amp_channels=[
            {"custom_channel_name": "A-000", "native_order": 0},
            {"custom_channel_name": "A-001", "native_order": 1},
        ],
        din_channels=[
            {"custom_channel_name": "DIN-00", "native_order": 0},
            {"custom_channel_name": "DIN-03", "native_order": 3},
        ],
    )

    data, _total, _dur = read_Intan_RHD2000_directory(tmp_path, header, "amp", 2, 0.0, float("inf"))
    n = data.shape[0]
    assert n >= num_samples - 1
    np.testing.assert_allclose(data[:, 0], ch2[:n].astype(np.float64) * 0.195, atol=1e-10)

    data_din, _, _ = read_Intan_RHD2000_directory(
        tmp_path, header, "din", [1, 2], 0.0, float("inf")
    )
    n_din = data_din.shape[0]
    assert n_din >= num_samples - 1
    expected_bit0 = np.array([1, 1, 1, 1], dtype=np.float64)
    expected_bit3 = np.array([0, 0, 1, 1], dtype=np.float64)
    np.testing.assert_array_equal(data_din[:, 0], expected_bit0[:n_din])
    np.testing.assert_array_equal(data_din[:, 1], expected_bit3[:n_din])

    data_t, _, _ = read_Intan_RHD2000_directory(tmp_path, header, "time", 1, 0.0, float("inf"))
    n_t = data_t.shape[0]
    expected_t = times.astype(np.float64) / sample_rate
    assert n_t >= num_samples - 1
    np.testing.assert_array_equal(data_t[:, 0], expected_t[:n_t])


def test_reader_detects_directory_from_prefixed_info(tmp_path: Path) -> None:
    reader = ndr_reader_intan__rhd()

    info_path = tmp_path / "febc0_u000_000_info.rhd"
    time_path = tmp_path / "febc0_u000_000_time.dat"
    info_path.touch()
    time_path.touch()

    _, _, isdirectory, _ = reader.filenamefromepochfiles([str(info_path), str(time_path)])
    assert isdirectory is True

    info_bare = tmp_path / "info.rhd"
    time_bare = tmp_path / "time.dat"
    info_bare.touch()
    time_bare.touch()
    _, _, isdirectory_bare, _ = reader.filenamefromepochfiles([str(info_bare), str(time_bare)])
    assert isdirectory_bare is True
