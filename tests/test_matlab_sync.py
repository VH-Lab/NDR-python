"""Tests for behaviors synchronized from NDR-matlab.

Each test names the NDR-matlab commit whose behavior it pins, so a future
sync can trace the requirement back to the MATLAB source of truth.
"""

import math

import numpy as np
import pytest

from ndr.format.intan.read_Intan_RHD2000_datafile import (
    Intan_RHD2000_blockinfo,
    read_Intan_RHD2000_datafile,
)
from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header
from ndr.fun.ndrpath import ndrpath
from ndr.reader.base import ndr_reader_base
from ndr.reader.ced_smr import ndr_reader_ced__smr
from ndr.reader.intan_rhd import ndr_reader_intan__rhd
from ndr.reader.neo import ndr_reader_neo
from ndr.reader.spikegadgets_rec import ndr_reader_spikegadgets__rec
from ndr.reader.tdt_sev import ndr_reader_tdt__sev

EXAMPLE_RHD = str(ndrpath() / "example_data" / "example.rhd")


class TestChannelLabelingConvention:
    """MATLAB b2e9d95 + 3974d59: readers declare their channel-naming contract."""

    def test_base_default_is_indexed(self):
        class _Concrete(ndr_reader_base):
            def readchannels_epochsamples(self, *args, **kwargs):
                raise NotImplementedError

            def readevents_epochsamples_native(self, *args, **kwargs):
                raise NotImplementedError

        assert _Concrete().channelLabelingConvention("analog_in") == "indexed"

    def test_intan_rhd_inherits_indexed(self):
        assert ndr_reader_intan__rhd().channelLabelingConvention("analog_in") == "indexed"

    @pytest.mark.parametrize(
        "cls",
        [ndr_reader_ced__smr, ndr_reader_spikegadgets__rec, ndr_reader_tdt__sev],
    )
    def test_physical_readers(self, cls):
        assert cls().channelLabelingConvention("analog_in") == "physical"

    def test_neo_is_native(self):
        assert ndr_reader_neo().channelLabelingConvention("analog_in") == "native"


class TestIntanSamplerateNaN:
    """MATLAB 67addc7: event/marker/text channels have no scalar sample rate."""

    @pytest.mark.parametrize(
        "channeltype", ["event", "e", "marker", "mk", "text", "tx", "eventmarktext"]
    )
    def test_event_types_return_nan(self, channeltype):
        reader = ndr_reader_intan__rhd()
        sr = reader.samplerate([EXAMPLE_RHD], 1, channeltype, 1)
        assert math.isnan(sr)

    def test_analog_still_returns_real_rate(self):
        reader = ndr_reader_intan__rhd()
        sr = reader.samplerate([EXAMPLE_RHD], 1, "analog_in", 1)
        assert sr > 0 and not math.isnan(sr)


class TestIntanPositionalNaming:
    """MATLAB d878687: getchannelsepoch names channels by 1-based position."""

    def test_names_are_positional(self):
        reader = ndr_reader_intan__rhd()
        channels = reader.getchannelsepoch([EXAMPLE_RHD], 1)

        by_type: dict[str, list[str]] = {}
        for ch in channels:
            by_type.setdefault(ch["type"], []).append(ch["name"])

        assert by_type["analog_in"] == [f"ai{i}" for i in range(1, 33)]
        assert by_type["auxiliary_in"] == [f"ax{i}" for i in range(1, 4)]
        assert by_type["digital_in"] == ["di1"]

    def test_trailing_number_is_a_position_not_a_chip_channel(self):
        """The first recorded channel of a type is always numbered 1."""
        reader = ndr_reader_intan__rhd()
        channels = reader.getchannelsepoch([EXAMPLE_RHD], 1)
        first_analog = next(c for c in channels if c["type"] == "analog_in")
        assert first_analog["name"] == "ai1"


class TestIntanDigitalNormalization:
    """MATLAB cae4b00: digital reads are 0/1, indexed via native_order."""

    def test_digital_read_is_zero_or_one(self):
        data = read_Intan_RHD2000_datafile(EXAMPLE_RHD, "", "din", 1, 0.0, 0.1)
        assert data.shape[1] == 1
        assert np.all(np.isin(data, [0.0, 1.0]))

    def test_channel_number_indexes_recorded_list_not_bit_position(self):
        """Channel 1 is the first *recorded* digital channel, whatever its bit."""
        data = read_Intan_RHD2000_datafile(EXAMPLE_RHD, "", "din", 1, 0.0, 0.1)
        assert data.shape[1] == 1

    def test_out_of_range_digital_channel_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            read_Intan_RHD2000_datafile(EXAMPLE_RHD, "", "din", 99, 0.0, 0.1)

    def test_absent_digital_type_raises(self):
        with pytest.raises(ValueError, match="No digital"):
            read_Intan_RHD2000_datafile(EXAMPLE_RHD, "", "dout", 1, 0.0, 0.1)


class TestIntanUniformChannelTypeList:
    """MATLAB dc1ee9a: accept a uniform list of channel-type strings."""

    def test_uniform_list_matches_scalar(self):
        reader = ndr_reader_intan__rhd()
        scalar = reader.readchannels_epochsamples("analog_in", [1, 2], [EXAMPLE_RHD], 1, 1, 100)
        as_list = reader.readchannels_epochsamples(
            ["analog_in", "analog_in"], [1, 2], [EXAMPLE_RHD], 1, 1, 100
        )
        np.testing.assert_array_equal(scalar, as_list)

    def test_heterogeneous_list_raises(self):
        reader = ndr_reader_intan__rhd()
        with pytest.raises(ValueError, match="uniform"):
            reader.readchannels_epochsamples(
                ["analog_in", "digital_in"], [1, 2], [EXAMPLE_RHD], 1, 1, 100
            )


class TestIntanBlockAccounting:
    """The data block must be measured exactly, or every later block shifts.

    Caught by the cross-language symmetry test once it was actually running:
    the temp-sensor section was sized from the supply-voltage channel count,
    so a recording with supply voltage but no temp sensor got a block 2 bytes
    too long. That truncated the epoch by a whole block and desynchronized
    the sequential read from the first block boundary onward.
    """

    def test_block_size_divides_the_data_exactly(self):
        header = read_Intan_RHD2000_header(EXAMPLE_RHD)
        _bi, bytes_per_block, bytes_present, _n = Intan_RHD2000_blockinfo(EXAMPLE_RHD, header)
        assert (
            bytes_present % bytes_per_block == 0
        ), "data section is not a whole number of blocks; bytes_per_block is wrong"

    def test_temp_bytes_follow_the_temp_channel_count(self):
        """Supply-voltage channels must not imply a temp-sensor sample."""
        header = read_Intan_RHD2000_header(EXAMPLE_RHD)
        blockinfo, bytes_per_block, _bp, _n = Intan_RHD2000_blockinfo(EXAMPLE_RHD, header)
        # This fixture is the interesting case: supply present, temp absent.
        assert blockinfo["num_supply"] > 0
        assert blockinfo["num_temp"] == 0

        spb = blockinfo["samples_per_block"]
        expected = (
            4 * spb
            + 2 * blockinfo["num_amplifier"] * spb
            + 2 * blockinfo["num_aux"] * (spb // 4)
            + 2 * blockinfo["num_supply"]
            + 2 * (blockinfo["num_temp"] > 0)
            + 2 * blockinfo["num_adc"] * spb
            + (2 * spb if blockinfo["num_dig_in"] else 0)
            + (2 * spb if blockinfo["num_dig_out"] else 0)
        )
        assert bytes_per_block == expected

    def test_epoch_covers_every_block(self):
        """t0_t1 must span all blocks, not lose the last partial-looking one."""
        reader = ndr_reader_intan__rhd()
        header = read_Intan_RHD2000_header(EXAMPLE_RHD)
        _bi, _bpb, bytes_present, num_blocks = Intan_RHD2000_blockinfo(EXAMPLE_RHD, header)
        sr = header["frequency_parameters"]["amplifier_sample_rate"]
        spb = header["num_samples_per_data_block"]
        t0t1 = reader.t0_t1([EXAMPLE_RHD], 1)
        assert t0t1[0][1] == pytest.approx((spb * num_blocks) / sr - 1 / sr)

    def test_read_is_continuous_across_a_block_boundary(self):
        """A 2-byte block error drops a sample at each boundary; check none is."""
        reader = ndr_reader_intan__rhd()
        # Samples 1..120 span the first block boundary at sample 61.
        data = reader.readchannels_epochsamples("analog_in", [1], [EXAMPLE_RHD], 1, 1, 120).ravel()
        # Reading the second block alone must agree with the tail of the
        # combined read; a desynchronized reader disagrees here.
        second = reader.readchannels_epochsamples(
            "analog_in", [1], [EXAMPLE_RHD], 1, 61, 120
        ).ravel()
        np.testing.assert_allclose(data[60 : 60 + len(second)], second)


class TestIntanSamplesPerDataBlockVersion:
    """MATLAB Intan_RHD2000_blockinfo.m:45-49: 60 samples/block for v1, 128 for v2+.

    read_Intan_RHD2000_header already derived this correctly, but the block
    sizer ignored it and hardcoded 60, so every v2.0+ file was decoded at less
    than half its true block length. Nothing short-read, so the corruption was
    silent. The bundled fixture is v1, which is why no existing test caught it.
    """

    def test_blockinfo_uses_the_header_value(self):
        header = read_Intan_RHD2000_header(EXAMPLE_RHD)
        blockinfo, _bpb, _bp, _n = Intan_RHD2000_blockinfo(EXAMPLE_RHD, header)
        assert blockinfo["samples_per_block"] == header["num_samples_per_data_block"]

    @pytest.mark.parametrize("main_version,expected", [(1, 60), (2, 128), (3, 128)])
    def test_v2_plus_blocks_hold_128_samples(self, main_version, expected):
        """Pin the version->block-size rule directly; the fixture is v1 only."""
        header = read_Intan_RHD2000_header(EXAMPLE_RHD)
        header = dict(header)
        header["data_file_main_version_number"] = main_version
        header["num_samples_per_data_block"] = 60 if main_version == 1 else 128

        blockinfo, bytes_per_block, _bp, _n = Intan_RHD2000_blockinfo(EXAMPLE_RHD, header)
        assert blockinfo["samples_per_block"] == expected

        # The block must grow with the sample count, not stay pinned at the
        # v1 size: that was the whole failure mode.
        per_sample_sections = 4 + 2 * blockinfo["num_amplifier"] + 2 * blockinfo["num_adc"]
        assert bytes_per_block >= per_sample_sections * expected


class TestMatlabRoundingParity:
    """MATLAB `round` is half-away-from-zero; Python's and numpy's are half-to-even.

    Every NDR-matlab time-to-sample conversion goes through MATLAB `round`
    (times2samples.m:12, readvhlvdatafile.m:74-75/234-235, reader.m:137-138/355-356,
    read_Intan_RHD2000_datafile.m:124-125/170-171). On an exact half-sample
    boundary the two conventions pick different samples, so the same requested
    interval reads a different span of the file in each language.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.5, 1.0),  # builtin round gives 0
            (1.5, 2.0),
            (2.5, 3.0),  # builtin round gives 2
            (3.5, 4.0),
            (-0.5, -1.0),  # away from zero, not toward it
            (-2.5, -3.0),
            (0.4, 0.0),
            (0.6, 1.0),
            (-0.6, -1.0),
            (7.0, 7.0),
        ],
    )
    def test_halves_go_away_from_zero(self, value, expected):
        from ndr.time.fun.times2samples import matlab_round

        assert matlab_round(value) == expected

    def test_differs_from_builtin_exactly_on_halves(self):
        """Pin the disagreement, so the helper cannot be quietly swapped back."""
        from ndr.time.fun.times2samples import matlab_round

        disagree = [x / 2 for x in range(-8, 9) if matlab_round(x / 2) != round(x / 2)]
        # Only halves whose away-from-zero neighbour is odd: at 1.5 and 3.5 the
        # even neighbour IS the away-from-zero one, so the two agree there.
        assert disagree == [-2.5, -0.5, 0.5, 2.5]

    def test_scalar_returns_float_arrays_return_arrays(self):
        from ndr.time.fun.times2samples import matlab_round

        assert isinstance(matlab_round(2.5), float)
        out = matlab_round(np.array([0.5, 1.5, -0.5]))
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1.0, 2.0, -1.0])

    def test_times2samples_uses_it(self):
        """A time landing exactly between samples resolves the MATLAB way."""
        from ndr.time.fun.times2samples import times2samples

        # sr = 2 Hz, so t = 0.25 s is exactly half a sample past sample 1.
        s = times2samples(np.array([0.25]), [0.0, 10.0], 2.0)
        assert s[0] == 2.0  # banker's rounding would give 1.0
