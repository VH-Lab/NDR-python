"""Reads must honour the caller's channel order and each channel's own stream.

Both bugs here return well-formed arrays of the right shape filled with real
recorded values -- just the wrong ones -- so nothing downstream can notice.
"""

import numpy as np
import pytest

from ndr.fun.ndrpath import ndrpath

EXAMPLE_ABF = str(ndrpath() / "example_data" / "example.abf")


class TestAxonChannelOrder:
    """The returned columns must follow the requested order, not sorted order."""

    def _reader(self):
        pytest.importorskip("pyabf")
        from ndr.reader.axon_abf import ndr_reader_axon__abf

        return ndr_reader_axon__abf(), [EXAMPLE_ABF]

    def _n_analog(self, reader, es):
        return sum(1 for c in reader.getchannelsepoch(es, 1) if c["type"] == "analog_in")

    def test_descending_request_is_not_silently_sorted(self):
        reader, es = self._reader()
        if self._n_analog(reader, es) < 2:
            pytest.skip("example.abf has fewer than 2 analog channels")

        ascending = reader.readchannels_epochsamples("analog_in", [1, 2], es, 1, 1, 200)
        descending = reader.readchannels_epochsamples("analog_in", [2, 1], es, 1, 1, 200)

        # Same data, columns transposed -- not the same array.
        np.testing.assert_array_equal(descending[:, 0], ascending[:, 1])
        np.testing.assert_array_equal(descending[:, 1], ascending[:, 0])

    def test_single_channel_reads_match_their_column_in_a_multi_read(self):
        """The strongest form: each column must be the channel that was asked for."""
        reader, es = self._reader()
        n = self._n_analog(reader, es)
        if n < 2:
            pytest.skip("example.abf has fewer than 2 analog channels")

        requested = list(range(min(n, 3), 0, -1))  # e.g. [3, 2, 1]
        combined = reader.readchannels_epochsamples("analog_in", requested, es, 1, 1, 200)

        for col, ch in enumerate(requested):
            alone = reader.readchannels_epochsamples("analog_in", [ch], es, 1, 1, 200)
            np.testing.assert_array_equal(
                combined[:, col], alone.ravel(), err_msg=f"column {col} is not channel {ch}"
            )


class TestAxonChannelOrderSynthetic:
    """Pin the ordering logic itself.

    The bundled example.abf has a single analog channel, so a fixture-based
    test can only skip -- and a skipping test would leave this fix unverified.
    A stub ABF with per-channel constant traces makes the permutation visible:
    channel N reads as a trace of N.
    """

    N_CHANNELS = 4
    N_POINTS = 50

    @pytest.fixture
    def stub_abf(self, monkeypatch):
        import importlib

        # `from ndr.format.axon import read_abf` binds the *function* (the
        # package re-exports it), so reach the module explicitly.
        read_abf_mod = importlib.import_module("ndr.format.axon.read_abf")

        outer = self

        class _StubABF:
            sweepCount = 1

            def __init__(self, filename):
                self._channel = 0

            def setSweep(self, sweep, channel=0):
                self._channel = channel

            @property
            def sweepY(self):
                # Channel index c (0-based) reads as a constant (c + 1).
                return np.full(outer.N_POINTS, self._channel + 1, dtype=np.float64)

        class _StubPyabf:
            ABF = _StubABF

        monkeypatch.setattr(read_abf_mod, "pyabf", _StubPyabf)
        monkeypatch.setattr(
            read_abf_mod, "read_abf_header", lambda _f: {"si": 100.0, "recTime": [0.0, 1000.0]}
        )
        return read_abf_mod.read_abf

    @pytest.mark.parametrize(
        "requested",
        [[1, 2, 3], [3, 2, 1], [2, 4, 1, 3], [4, 1], [2], [3, 1]],
    )
    def test_columns_follow_the_requested_order(self, stub_abf, requested):
        data = stub_abf("stub.abf", channel_type="ai", channel_numbers=requested, t0=0.0, t1=1.0)
        assert data.shape[1] == len(requested)
        for col, ch in enumerate(requested):
            assert set(np.unique(data[:, col])) == {float(ch)}, (
                f"column {col} holds channel {int(data[0, col])}, expected {ch}"
            )


class TestCedStreamResolution:
    """Sample count, rate and time base must come from the channel's own stream."""

    def _reader(self):
        pytest.importorskip("neo")
        smr = ndrpath() / "example_data" / "example.smr"
        if not smr.exists():
            pytest.skip("example.smr not available")
        from ndr.reader.ced_smr import ndr_reader_ced__smr

        return ndr_reader_ced__smr(), [str(smr)]

    def test_time_channel_matches_the_channel_it_was_asked_about(self):
        """A 'time' read must use the requested channel's own sampling."""
        reader, es = self._reader()
        channels = [c for c in reader.getchannelsepoch(es, 1) if c["type"] == "analog_in"]
        if len(channels) < 1:
            pytest.skip("example.smr has no analog channels")

        ch_num = int(channels[0]["name"].lstrip("abcdefghijklmnopqrstuvwxyz_"))
        t = reader.readchannels_epochsamples("time", [ch_num], es, 1, 1, 50).ravel()
        sr = reader.samplerate(es, 1, "analog_in", ch_num)

        assert len(t) == 50
        # Consecutive time steps must equal 1/sr for THIS channel.
        np.testing.assert_allclose(np.diff(t), 1.0 / sr, rtol=1e-6)
