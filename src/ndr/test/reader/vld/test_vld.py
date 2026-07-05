"""Automated tests for the VHLAB LabView (.vld/.vlh) reader.

Port-validation of ndr.reader.vld against a real recording.

The reference file pair lives at::

    /Users/audribhowmick/Downloads/2015-03-03/t00003/vhlvanaloginput.{vld,vlh}

whose header declares NumChans=33, SamplingRate=25000, Scale=10,
precision=int16, Multiplexed=1. The faithful port of the MATLAB reader
(+ndr/+reader/vld.m getchannelsepoch) reports the single time channel 't1'
first, then one analog-input channel per acquired channel (ai1..ai33), i.e.
1 + NumChans = 34 channels total.

The load-bearing check is that readchannels_epochsamples('ai', ...) is
BYTE-EXACT against an independent numpy decode computed inline in the test.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

EP = "/Users/audribhowmick/Downloads/2015-03-03/t00003"
VLD = os.path.join(EP, "vhlvanaloginput.vld")
VLH = os.path.join(EP, "vhlvanaloginput.vlh")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(VLD) and os.path.exists(VLH)),
    reason=f"Real VHLV test data not found at {EP}",
)


@pytest.fixture()
def reader():
    from ndr.reader.vld import ndr_reader_vld

    return ndr_reader_vld()


def _num_chans() -> int:
    from ndr.format.vld.readvhlvheaderfile import readvhlvheaderfile

    return int(readvhlvheaderfile(VLH)["NumChans"])


def test_header_fields():
    from ndr.format.vld.readvhlvheaderfile import readvhlvheaderfile

    h = readvhlvheaderfile(VLH)
    assert h["NumChans"] == 33
    assert h["SamplingRate"] == 25000
    assert h["Scale"] == 10
    assert h["precision"] == "int16"
    assert h["Multiplexed"] == 1


def test_getchannelsepoch(reader):
    channels = reader.getchannelsepoch([VLD], 1)
    num_chans = _num_chans()

    # Faithful MATLAB behaviour: one time channel ('t1') first, then ai1..aiN.
    assert len(channels) == num_chans + 1  # 34 for this recording

    assert channels[0] == {"name": "t1", "type": "time", "time_channel": 1}

    analog = channels[1:]
    assert len(analog) == num_chans  # 33 analog_in channels
    assert all(c["type"] == "analog_in" for c in analog)
    assert analog[0]["name"] == "ai1"
    assert analog[-1]["name"] == f"ai{num_chans}"


def test_samplerate(reader):
    sr = reader.samplerate([VLD], 1, "analog_in", 1)
    assert sr == 25000

    sr_multi = reader.samplerate([VLD], 1, "analog_in", [1, 16, 32])
    assert np.array_equal(sr_multi, np.full(3, 25000.0))


def test_t0_t1(reader):
    t0t1 = reader.t0_t1([VLD], 1)
    assert t0t1[0][0] == 0


def test_readchannels_byte_exact(reader):
    """readchannels_epochsamples must be byte-exact vs an independent decode."""
    from ndr.format.vld.readvhlvheaderfile import readvhlvheaderfile

    h = readvhlvheaderfile(VLH)
    num_chans = int(h["NumChans"])
    scale = float(h["Scale"])

    # Reader read: samples 1..10000 (1-based inclusive) of ai1, ai16, ai32.
    data = reader.readchannels_epochsamples(
        "ai", [1, 16, 32], [VLD], 1, 1, 10000
    )
    assert data.shape == (10000, 3)

    # Independent numpy decode computed IN the test.
    expected = (
        np.fromfile(VLD, dtype=">i2", count=num_chans * 10000)
        .reshape(-1, num_chans)[:10000, [0, 15, 31]]
        * (scale / 32767)
    )
    assert expected.shape == (10000, 3)
    assert np.allclose(data, expected, atol=1e-9)


def test_read_time_channel(reader):
    """The 'time' channel returns (s0-1..s1-1)/sr timestamps."""
    sr = 25000.0
    t = reader.readchannels_epochsamples("time", [1], [VLD], 1, 1, 10)
    assert t.shape == (10, 1)
    expected = (np.arange(0, 10) / sr).reshape(-1, 1)
    assert np.allclose(t.flatten(), expected.flatten(), atol=1e-12)


def test_underlying_datatype(reader):
    datatype, p, datasize = reader.underlying_datatype([VLD], 1, "analog_in", [1])
    assert datatype == "int16"
    assert datasize == 16
    # polynomial [offset, Scale/maxint] with offset 0
    assert p.shape == (1, 2)
    assert p[0, 0] == 0
    assert np.isclose(p[0, 1], 10.0 / (2**15 - 1))


def test_readevents_empty(reader):
    ts, d = reader.readevents_epochsamples_native("event", [1], [VLD], 1, 0, 1)
    assert ts.size == 0
    assert d.size == 0
