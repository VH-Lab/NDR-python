"""Unit tests for NDR readers."""

import numpy as np
import pytest

from ndr.known_readers import known_readers
from ndr.reader.intan_rhd import IntanRHD
from ndr.string.channelstring2channels import channelstring2channels
from ndr.string.str2intseq import str2intseq
from ndr.time.fun.samples2times import samples2times
from ndr.time.fun.times2samples import times2samples


class TestIntanRHDStatic:
    """Test static methods of IntanRHD that don't require data files."""

    def test_mfdaqchanneltype2intanheadertype(self):
        assert IntanRHD.mfdaqchanneltype2intanheadertype("analog_in") == "amplifier_channels"
        assert IntanRHD.mfdaqchanneltype2intanheadertype("ai") == "amplifier_channels"
        assert IntanRHD.mfdaqchanneltype2intanheadertype("digital_in") == "board_dig_in_channels"

    def test_intanheadertype2mfdaqchanneltype(self):
        assert IntanRHD.intanheadertype2mfdaqchanneltype("amplifier_channels") == "analog_in"
        assert IntanRHD.intanheadertype2mfdaqchanneltype("board_dig_in_channels") == "digital_in"

    def test_mfdaqchanneltype2intanchanneltype(self):
        assert IntanRHD.mfdaqchanneltype2intanchanneltype("analog_in") == "amp"
        assert IntanRHD.mfdaqchanneltype2intanchanneltype("digital_in") == "din"
        assert IntanRHD.mfdaqchanneltype2intanchanneltype("time") == "time"

    def test_intanchanneltype2mfdaqchanneltype(self):
        assert IntanRHD.intanchanneltype2mfdaqchanneltype("amp") == "ai"
        assert IntanRHD.intanchanneltype2mfdaqchanneltype("din") == "di"

    def test_intananychannelname2intanchanneltype(self):
        ct, absolute = IntanRHD.intananychannelname2intanchanneltype("ai")
        assert ct == "amp"
        assert absolute is False

        ct, absolute = IntanRHD.intananychannelname2intanchanneltype("A")
        assert ct == "amp"
        assert absolute is True

    def test_mfdaqchanneltype2intanfreqheader(self):
        assert IntanRHD.mfdaqchanneltype2intanfreqheader("ai") == "amplifier_sample_rate"
        assert IntanRHD.mfdaqchanneltype2intanfreqheader("auxiliary") == "aux_input_sample_rate"

    def test_intanname2mfdaqname_struct(self):
        name = IntanRHD.intanname2mfdaqname(
            "analog_in",
            {"native_channel_name": "A-000", "chip_channel": 0},
        )
        assert name == "ai1"

    def test_intanname2mfdaqname_string(self):
        name = IntanRHD.intanname2mfdaqname("analog_in", "A-000")
        assert name == "ai1"

    def test_intanname2mfdaqname_aux(self):
        name = IntanRHD.intanname2mfdaqname(
            "auxiliary_in",
            {"native_channel_name": "AUX1", "chip_channel": 0},
        )
        assert name == "ax1"

    def test_filenamefromepochfiles(self):
        reader = IntanRHD()
        fn, pdir, isdir = reader.filenamefromepochfiles(["/path/to/data.rhd"])
        assert fn == "/path/to/data.rhd"
        assert isdir is False

    def test_filenamefromepochfiles_no_rhd(self):
        reader = IntanRHD()
        with pytest.raises(ValueError, match="Need 1 .rhd"):
            reader.filenamefromepochfiles(["/path/to/data.abf"])

    def test_filenamefromepochfiles_multiple_rhd(self):
        reader = IntanRHD()
        with pytest.raises(ValueError, match="Need only 1"):
            reader.filenamefromepochfiles(["/path/a.rhd", "/path/b.rhd"])


class TestKnownReaders:
    def test_known_readers(self):
        readers = known_readers()
        assert len(readers) > 0
        # Check that intan is in there
        flat = [t for entry in readers for t in entry]
        assert "intan" in flat
        assert "rhd" in flat
        assert "abf" in flat


class TestChannelString:
    def test_simple(self):
        prefix, numbers = channelstring2channels("ai1")
        assert prefix == ["ai"]
        assert numbers == [1]

    def test_range(self):
        prefix, numbers = channelstring2channels("ai1-3")
        assert prefix == ["ai", "ai", "ai"]
        assert numbers == [1, 2, 3]

    def test_comma(self):
        prefix, numbers = channelstring2channels("ai1,3,5")
        assert prefix == ["ai", "ai", "ai"]
        assert numbers == [1, 3, 5]

    def test_mixed(self):
        prefix, numbers = channelstring2channels("ai1-3+di2-4")
        assert prefix == ["ai", "ai", "ai", "di", "di", "di"]
        assert numbers == [1, 2, 3, 2, 3, 4]

    def test_no_numbers(self):
        with pytest.raises(ValueError, match="No numbers"):
            channelstring2channels("ai")


class TestStr2IntSeq:
    def test_single(self):
        assert str2intseq("5") == [5]

    def test_range(self):
        assert str2intseq("1-3") == [1, 2, 3]

    def test_comma(self):
        assert str2intseq("1,3,5") == [1, 3, 5]

    def test_mixed(self):
        assert str2intseq("1,3-5,2") == [1, 3, 4, 5, 2]

    def test_empty(self):
        assert str2intseq("") == []


class TestTimeFunctions:
    def test_samples2times(self):
        t = samples2times(np.array([1, 2, 3]), (0.0, 1.0), 10.0)
        np.testing.assert_allclose(t, [0.0, 0.1, 0.2])

    def test_times2samples(self):
        s = times2samples(np.array([0.0, 0.1, 0.2]), (0.0, 1.0), 10.0)
        np.testing.assert_allclose(s, [1, 2, 3])

    def test_samples2times_inf(self):
        t = samples2times(np.array([-np.inf, np.inf]), (0.0, 1.0), 10.0)
        assert t[0] == 0.0
        assert t[1] == 1.0

    def test_times2samples_inf(self):
        s = times2samples(np.array([-np.inf, np.inf]), (0.0, 1.0), 10.0)
        assert s[0] == 1
        assert s[1] == 1 + 10.0 * 1.0


class TestReaderWrapper:
    def test_import(self):
        import ndr

        assert hasattr(ndr, "reader")

    def test_unknown_type(self):
        from ndr.reader_wrapper import Reader

        with pytest.raises(ValueError, match="Do not know"):
            Reader("nonexistent_format")
