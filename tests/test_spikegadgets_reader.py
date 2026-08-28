"""Tests for the SpikeGadgets .rec reader class.

The reader is thin plumbing over ndr.format.spikegadgets: its job is to build
the channel table, translate NDR channel numbers into the byte/bit addressing
the format functions expect, and get the timing right. These pin that
translation against the real example.rec, since a wrong mapping reads a
different channel and returns entirely plausible data.
"""

import numpy as np
import pytest

from ndr.format.spikegadgets.read_rec_config import read_rec_config
from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels
from ndr.fun.ndrpath import ndrpath
from ndr.reader.spikegadgets_rec import ndr_reader_spikegadgets__rec

REC = ndrpath() / "example_data" / "example.rec"


@pytest.fixture(scope="module")
def reader():
    if not REC.exists():
        pytest.skip("example.rec not available")
    return ndr_reader_spikegadgets__rec()


@pytest.fixture(scope="module")
def es():
    return [str(REC)]


@pytest.fixture(scope="module")
def config():
    cfg, _channels = read_rec_config(str(REC))
    return cfg


class TestChannelTable:
    def test_time_channel_comes_first(self, reader, es):
        channels = reader.getchannelsepoch(es, 1)
        assert channels[0] == {"name": "t1", "type": "time", "time_channel": 1}

    def test_one_analog_in_per_ntrode_channel(self, reader, es, config):
        channels = reader.getchannelsepoch(es, 1)
        analog = [c for c in channels if c["type"] == "analog_in"]
        assert len(analog) == len(config["nTrodes"]) * 4
        # numChannels counts exactly the nTrode channels.
        assert len(analog) == int(config["numChannels"])

    def test_names_follow_the_ndr_prefixes(self, reader, es):
        channels = reader.getchannelsepoch(es, 1)
        prefixes = {
            "analog_in": "ai",
            "auxiliary": "ax",
            "digital_in": "di",
            "digital_out": "do",
        }
        for c in channels:
            if c["type"] in prefixes:
                assert c["name"].startswith(prefixes[c["type"]]), c

    def test_sorted_by_type_then_number(self, reader, es):
        channels = reader.getchannelsepoch(es, 1)[1:]  # skip the time channel
        keyed = [
            (c["type"], int("".join(ch for ch in c["name"] if ch.isdigit()))) for c in channels
        ]
        assert keyed == sorted(keyed)

    def test_addressing_fields_are_not_exposed(self, reader, es):
        """getchannelsepoch.m strips startbyte/bit/number before returning."""
        for c in reader.getchannelsepoch(es, 1):
            assert set(c) == {"name", "type", "time_channel"}

    def test_mcu_inputs_are_offset_past_the_direct_inputs(self, reader, es):
        """MCU_Din%d becomes di(%d + 32), so it cannot collide with Din%d."""
        names = [c["name"] for c in reader.getchannelsepoch(es, 1) if c["type"] == "digital_in"]
        assert len(names) == len(set(names)), "digital_in names must be unique"
        numbers = sorted(int(n[2:]) for n in names)
        assert max(numbers) > 32, "expected MCU inputs above the 32 direct inputs"


class TestTiming:
    def test_samplerate_matches_the_config(self, reader, es, config):
        assert reader.samplerate(es, 1) == float(config["samplingRate"])

    def test_t0_t1_spans_a_whole_number_of_packets(self, reader, es, config):
        """The epoch must end on a real sample.

        spikegadgets_rec.m sizes the packet as headerSize + 2 + channels and
        subtracts only headerSize from the file length, which yields 60090.7
        packets on this fixture -- a fractional sample count -- and an epoch
        about 17 ms too long. This checks the corrected arithmetic instead.
        """
        sr = float(config["samplingRate"])
        t0, t1 = reader.t0_t1(es, 1)[0]
        assert t0 == 0.0

        n_samples = round(t1 * sr) + 1
        assert n_samples == pytest.approx(t1 * sr + 1, abs=1e-6), "t1 must land on a sample"

        header_bytes = int(config["headerSize"]) * 2
        channel_bytes = int(config["numChannels"]) * 2
        from ndr.format.spikegadgets.read_rec_configsize import read_rec_configsize

        data_bytes = REC.stat().st_size - read_rec_configsize(REC)
        assert n_samples == data_bytes // (header_bytes + 4 + channel_bytes)

    def test_t0_t1_is_not_the_old_matlab_value(self, reader, es, config):
        """Guard the divergence, so a future 'sync' does not reintroduce it."""
        sr = float(config["samplingRate"])
        header_bytes = int(config["headerSize"]) * 2
        channel_bytes = int(config["numChannels"]) * 2
        matlab_blocks = (REC.stat().st_size - header_bytes) / (header_bytes + 2 + channel_bytes)
        matlab_t1 = (matlab_blocks - 1) / sr

        _t0, t1 = reader.t0_t1(es, 1)[0]
        assert abs(t1 - matlab_t1) > 1e-3, "t0_t1 should not reproduce the old MATLAB value"

    def test_epochclock(self, reader, es):
        assert [c.type for c in reader.epochclock(es, 1)] == ["dev_local_time"]


class TestReads:
    def test_analog_in_matches_the_format_layer(self, reader, es, config):
        """Reader channel N is packet position N-1, as in the MATLAB version."""
        direct, _ts = read_rec_trodeChannels(
            str(REC),
            int(config["numChannels"]),
            [0],
            float(config["samplingRate"]),
            int(config["headerSize"]),
            1,
            100,
        )
        via_reader = reader.readchannels_epochsamples("analog_in", [1], es, 1, 1, 100)
        np.testing.assert_allclose(via_reader.ravel(), np.asarray(direct).ravel())

    def test_requested_channel_order_is_preserved(self, reader, es):
        ascending = reader.readchannels_epochsamples("analog_in", [1, 2], es, 1, 1, 100)
        descending = reader.readchannels_epochsamples("analog_in", [2, 1], es, 1, 1, 100)
        np.testing.assert_array_equal(descending[:, 0], ascending[:, 1])
        np.testing.assert_array_equal(descending[:, 1], ascending[:, 0])

    def test_time_channel_is_uniformly_spaced(self, reader, es, config):
        t = reader.readchannels_epochsamples("time", [1], es, 1, 1, 200).ravel()
        assert t.shape == (200,)
        np.testing.assert_allclose(np.diff(t), 1.0 / float(config["samplingRate"]), atol=1e-11)

    def test_auxiliary_read(self, reader, es):
        """MATLAB's auxiliary branch raises before reading; this one works."""
        aux = [c for c in reader.getchannelsepoch(es, 1) if c["type"] == "auxiliary"]
        number = int("".join(ch for ch in aux[0]["name"] if ch.isdigit()))
        data = reader.readchannels_epochsamples("auxiliary", [number], es, 1, 1, 100)
        assert data.shape == (100, 1)

    def test_digital_read_is_boolean(self, reader, es):
        din = [c for c in reader.getchannelsepoch(es, 1) if c["type"] == "digital_in"]
        number = int(din[0]["name"][2:])
        data = reader.readchannels_epochsamples("digital_in", [number], es, 1, 1, 100)
        assert data.shape == (100, 1)
        assert set(np.unique(data).tolist()) <= {False, True}

    def test_unknown_channel_number_raises(self, reader, es):
        with pytest.raises(ValueError, match="not recorded"):
            reader.readchannels_epochsamples("digital_in", [9999], es, 1, 1, 10)

    def test_unknown_channel_type_raises(self, reader, es):
        with pytest.raises(ValueError, match="Unknown channel type"):
            reader.readchannels_epochsamples("banana", [1], es, 1, 1, 10)


class TestEpochFiles:
    def test_no_rec_file_raises(self, reader):
        with pytest.raises(ValueError, match="Need 1 .rec file"):
            reader.filenamefromepochfiles(["a.txt", "b.dat"])

    def test_two_rec_files_raise(self, reader):
        with pytest.raises(ValueError, match="only 1"):
            reader.filenamefromepochfiles(["a.rec", "b.rec"])

    def test_labeling_convention(self, reader):
        assert reader.channelLabelingConvention("analog_in") == "physical"
