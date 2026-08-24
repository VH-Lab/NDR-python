"""Regression tests for the Intan RHD block-size computation.

Covers three fixes:
  * block sized from the header (60 for v1.x, 128 for v2.0+), not a literal 60;
  * the temperature-sensor block sized from num_temp_sensor_channels (0 or 2
    bytes total), not per supply-voltage channel;
  * digital channels bounds-checked against the recorded digital-channel list.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# NB: the intan package re-exports the function under this same dotted name, so a
# plain `import ... as mod` binds the function, not the module. Fetch the module
# object explicitly so monkeypatch can target its _get_header_size helper.
mod = importlib.import_module("ndr.format.intan.read_Intan_RHD2000_datafile")
Intan_RHD2000_blockinfo = mod.Intan_RHD2000_blockinfo
read_Intan_RHD2000_datafile = mod.read_Intan_RHD2000_datafile

EXAMPLE_RHD = Path(__file__).parents[1] / "src" / "ndr" / "example_data" / "example.rhd"


def _synthetic_header(
    *,
    main_version: int = 2,
    num_amp: int = 32,
    num_aux: int = 0,
    num_supply: int = 0,
    num_adc: int = 0,
    num_dig_in: int = 0,
    num_dig_out: int = 0,
    num_temp: int = 0,
) -> dict:
    spb = 60 if main_version == 1 else 128
    return {
        "amplifier_channels": [{} for _ in range(num_amp)],
        "aux_input_channels": [{} for _ in range(num_aux)],
        "supply_voltage_channels": [{} for _ in range(num_supply)],
        "board_adc_channels": [{} for _ in range(num_adc)],
        "board_dig_in_channels": [{"native_order": i} for i in range(num_dig_in)],
        "board_dig_out_channels": [{"native_order": i} for i in range(num_dig_out)],
        "num_temp_sensor_channels": num_temp,
        "dc_amplifier_data_saved": 0,
        "num_samples_per_data_block": spb,
    }


def test_v2_block_is_128_samples_and_8704_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_get_header_size", lambda f, h: 0)
    f = tmp_path / "fake.rhd"
    f.write_bytes(b"\x00" * (8704 * 3))
    header = _synthetic_header(main_version=2, num_amp=32)
    blockinfo, bytes_per_block, _bytes_present, num_data_blocks = Intan_RHD2000_blockinfo(f, header)
    assert blockinfo["samples_per_block"] == 128
    # 128*4 timestamps + 128*2*32 amplifier = 512 + 8192 = 8704
    assert bytes_per_block == 8704
    assert num_data_blocks == 3


def test_v1_block_is_60_samples_and_4080_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_get_header_size", lambda f, h: 0)
    f = tmp_path / "fake.rhd"
    f.write_bytes(b"\x00" * 4080)
    header = _synthetic_header(main_version=1, num_amp=32)
    blockinfo, bytes_per_block, _bp, _ndb = Intan_RHD2000_blockinfo(f, header)
    assert blockinfo["samples_per_block"] == 60
    # 60*4 + 60*2*32 = 240 + 3840 = 4080
    assert bytes_per_block == 4080


def test_temp_block_sized_from_temp_count_not_supply(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_get_header_size", lambda f, h: 0)
    f = tmp_path / "fake.rhd"
    f.write_bytes(b"\x00" * 100000)

    # 4 supply channels, 0 temp sensors -> temp contributes 0 bytes (not 8).
    h0 = _synthetic_header(main_version=1, num_amp=0, num_supply=4, num_temp=0)
    _, bpb_no_temp, _, _ = Intan_RHD2000_blockinfo(f, h0)

    # Same, but 1 temp sensor -> temp contributes exactly one uint16 (2 bytes).
    h1 = _synthetic_header(main_version=1, num_amp=0, num_supply=4, num_temp=1)
    _, bpb_one_temp, _, _ = Intan_RHD2000_blockinfo(f, h1)

    assert bpb_one_temp - bpb_no_temp == 2
    # Explicit: 4 supply channels add 8 bytes, temp adds 0. timestamps 60*4=240.
    assert bpb_no_temp == 240 + 2 * 4  # 248


@pytest.mark.skipif(not EXAMPLE_RHD.exists(), reason="example.rhd not available")
def test_example_rhd_block_divides_data_region_exactly():
    from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header

    header = read_Intan_RHD2000_header(EXAMPLE_RHD)
    _blockinfo, bytes_per_block, bytes_present, num_data_blocks = Intan_RHD2000_blockinfo(
        EXAMPLE_RHD, header
    )
    # A correct block size divides the data region with no remainder. The old
    # temp-sizing bug (charging temp bytes per supply channel) left a remainder.
    assert bytes_present % bytes_per_block == 0
    assert bytes_present == bytes_per_block * num_data_blocks


@pytest.mark.skipif(not EXAMPLE_RHD.exists(), reason="example.rhd not available")
def test_digital_channel_bounds_check():
    # example.rhd records exactly one digital-in channel; requesting a second
    # must raise rather than silently return an all-zero column.
    d1 = read_Intan_RHD2000_datafile(EXAMPLE_RHD, "", "din", [1], 0.0, 0.01)
    assert d1.shape[1] == 1
    with pytest.raises(ValueError):
        read_Intan_RHD2000_datafile(EXAMPLE_RHD, "", "din", [2], 0.0, 0.01)
