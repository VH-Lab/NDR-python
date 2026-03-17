"""Unit tests for ndr.reader.base."""

import numpy as np
import pytest

from ndr.reader.base import Base
from ndr.time.clocktype import ClockType


class ConcreteReader(Base):
    """Minimal concrete implementation for testing."""

    def readchannels_epochsamples(self, channeltype, channel, epochstreams, epoch_select, s0, s1):
        return np.zeros((s1 - s0 + 1, 1))

    def readevents_epochsamples_native(
        self, channeltype, channel, epochstreams, epoch_select, t0, t1
    ):
        return np.array([]), np.array([])


class TestBase:
    def test_mfdaq_channeltypes(self):
        ct = Base.mfdaq_channeltypes()
        assert "analog_in" in ct
        assert "digital_in" in ct
        assert "time" in ct
        assert len(ct) == 8

    def test_mfdaq_prefix(self):
        assert Base.mfdaq_prefix("analog_in") == "ai"
        assert Base.mfdaq_prefix("ai") == "ai"
        assert Base.mfdaq_prefix("digital_in") == "di"
        assert Base.mfdaq_prefix("time") == "t"
        assert Base.mfdaq_prefix("auxiliary") == "ax"
        assert Base.mfdaq_prefix("event") == "e"
        assert Base.mfdaq_prefix("marker") == "mk"

    def test_mfdaq_prefix_unknown(self):
        with pytest.raises(ValueError, match="Unknown channel type"):
            Base.mfdaq_prefix("nonexistent")

    def test_mfdaq_type(self):
        assert Base.mfdaq_type("ai") == "analog_in"
        assert Base.mfdaq_type("analog_in") == "analog_in"
        assert Base.mfdaq_type("di") == "digital_in"
        assert Base.mfdaq_type("time") == "time"

    def test_mfdaq_type_unknown(self):
        with pytest.raises(ValueError, match="unknown"):
            Base.mfdaq_type("nonexistent")

    def test_canbereadtogether_same_sr(self):
        reader = ConcreteReader()
        channels = [
            {"samplerate": 20000.0},
            {"samplerate": 20000.0},
        ]
        b, msg = reader.canbereadtogether(channels)
        assert b is True
        assert msg == ""

    def test_canbereadtogether_different_sr(self):
        reader = ConcreteReader()
        channels = [
            {"samplerate": 20000.0},
            {"samplerate": 10000.0},
        ]
        b, msg = reader.canbereadtogether(channels)
        assert b is False
        assert "same" in msg.lower()

    def test_canbereadtogether_all_nan(self):
        reader = ConcreteReader()
        channels = [
            {"samplerate": float("nan")},
            {"samplerate": float("nan")},
        ]
        b, msg = reader.canbereadtogether(channels)
        assert b is True

    def test_canbereadtogether_mixed_nan(self):
        reader = ConcreteReader()
        channels = [
            {"samplerate": 20000.0},
            {"samplerate": float("nan")},
        ]
        b, msg = reader.canbereadtogether(channels)
        assert b is False

    def test_epochclock_default(self):
        reader = ConcreteReader()
        ec = reader.epochclock(["test.rhd"])
        assert len(ec) == 1
        assert isinstance(ec[0], ClockType)
        assert ec[0].type == "dev_local_time"

    def test_t0_t1_default(self):
        reader = ConcreteReader()
        t = reader.t0_t1(["test.rhd"])
        assert len(t) == 1
        assert np.isnan(t[0][0])
        assert np.isnan(t[0][1])

    def test_getchannelsepoch_default(self):
        reader = ConcreteReader()
        channels = reader.getchannelsepoch(["test.rhd"])
        assert channels == []

    def test_underlying_datatype(self):
        reader = ConcreteReader()
        dt, p, ds = reader.underlying_datatype(["test.rhd"], 1, "analog_in", 1)
        assert dt == "float64"
        assert ds == 64
        assert p.shape == (1, 2)

    def test_underlying_datatype_unknown(self):
        reader = ConcreteReader()
        with pytest.raises(ValueError, match="Unknown channel type"):
            reader.underlying_datatype(["test.rhd"], 1, "nonexistent", 1)

    def test_might_have_time_gaps(self):
        reader = ConcreteReader()
        assert reader.MightHaveTimeGaps is False


class TestClockType:
    def test_init_default(self):
        ct = ClockType()
        assert ct.type == "dev_local_time"

    def test_init_custom(self):
        ct = ClockType("utc")
        assert ct.type == "utc"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown clock type"):
            ClockType("invalid_type")

    def test_needsepoch(self):
        assert ClockType("dev_local_time").needsepoch() is True
        assert ClockType("utc").needsepoch() is False

    def test_equality(self):
        a = ClockType("utc")
        b = ClockType("utc")
        c = ClockType("dev_local_time")
        assert a == b
        assert a != c

    def test_str_repr(self):
        ct = ClockType("utc")
        assert str(ct) == "utc"
        assert "utc" in repr(ct)
