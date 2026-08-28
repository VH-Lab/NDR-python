"""Exercise the CED sonpipe bridge against a stand-in CLI.

CED's sonpy cannot be installed on Linux or macOS for CPython 3.10-3.13, so the
real CLI cannot run in CI on the Python versions this package supports. Running
the bridge against tests/fake_sonpipe.py covers everything on our side of the
process boundary -- argv construction, JSON and binary parsing, kind dispatch,
sample-window arithmetic, and the crash/truncation checks -- without it.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ndr.format.ced import sonpipe
from ndr.format.ced.read_SOMSMR_datafile import read_SOMSMR_datafile
from ndr.format.ced.read_SOMSMR_header import read_SOMSMR_header
from ndr.format.ced.read_SOMSMR_sampleinterval import read_SOMSMR_sampleinterval

FAKE = Path(__file__).parent / "fake_sonpipe.py"
SR = 1000.0
N = 500


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """Point the bridge at the stand-in CLI and hand back a dummy file path."""
    monkeypatch.setenv("SONPIPE", shlex.join([sys.executable, str(FAKE)]))
    monkeypatch.delenv("FAKE_SONPIPE_FAULT", raising=False)
    sonpipe.reset_cache()
    yield str(tmp_path / "recording.smrx")
    sonpipe.reset_cache()


@pytest.fixture
def fault(monkeypatch):
    def _set(name):
        monkeypatch.setenv("FAKE_SONPIPE_FAULT", name)

    return _set


class TestExecutableDiscovery:
    def test_env_var_wins(self, fake_cli):
        assert sonpipe.executable() == [sys.executable, str(FAKE)]

    def test_result_is_cached(self, fake_cli, monkeypatch):
        first = sonpipe.executable()
        monkeypatch.setenv("SONPIPE", "/nonexistent/sonpipe")
        assert sonpipe.executable() == first, "lookup should not re-run once cached"

    def test_explicit_set_overrides(self, fake_cli):
        sonpipe.executable("/opt/venv/bin/sonpipe")
        assert sonpipe.executable() == ["/opt/venv/bin/sonpipe"]

    def test_not_found_raises(self, monkeypatch):
        """Every candidate failing must raise, with the remedy in the message.

        Clearing SONPIPE and PATH is not enough to force that: the
        `sys.executable -m sonpipe` candidate uses an absolute interpreter path,
        so wherever sonpipe is installed alongside the running Python -- as in
        the ced-integration job -- discovery still succeeds. That is correct
        behaviour, and it made this test pass only where sonpipe was absent.
        Failing the probe itself is what the not-found path actually depends on.
        """
        import importlib

        executable_mod = importlib.import_module("ndr.format.ced.sonpipe.executable")
        sonpipe.reset_cache()
        monkeypatch.setattr(executable_mod, "_works", lambda argv: False)
        try:
            with pytest.raises(sonpipe.SonpipeNotFoundError, match="Could not locate"):
                sonpipe.executable()
        finally:
            sonpipe.reset_cache()

    def test_every_candidate_is_probed_before_giving_up(self, monkeypatch):
        """The documented lookup order must actually be tried, in order."""
        import importlib

        executable_mod = importlib.import_module("ndr.format.ced.sonpipe.executable")
        sonpipe.reset_cache()
        monkeypatch.setenv("SONPIPE", "/nonexistent/sonpipe")
        tried = []

        def record(argv):
            tried.append(argv)
            return False

        monkeypatch.setattr(executable_mod, "_works", record)
        try:
            with pytest.raises(sonpipe.SonpipeNotFoundError):
                sonpipe.executable()
        finally:
            sonpipe.reset_cache()

        assert tried[0] == ["/nonexistent/sonpipe"], "SONPIPE must be tried first"
        assert ["sonpipe"] in tried, "the bare command must be tried"
        assert [sys.executable, "-m", "sonpipe"] in tried
        assert tried[-1] == ["python", "-m", "sonpipe"], "python -m is the last resort"

    def test_a_path_with_spaces_needs_no_escaping(self, monkeypatch, tmp_path):
        """argv is a list, so the shell never sees the path."""
        spaced = tmp_path / "some dir" / "fake_sonpipe.py"
        spaced.parent.mkdir()
        spaced.write_text(FAKE.read_text())
        sonpipe.reset_cache()
        monkeypatch.setenv("SONPIPE", shlex.join([sys.executable, str(spaced)]))
        try:
            assert sonpipe.executable() == [sys.executable, str(spaced)]
        finally:
            sonpipe.reset_cache()


class TestHeader:
    def test_channel_list(self, fake_cli):
        h = read_SOMSMR_header(fake_cli)
        assert [c["number"] for c in h["channelinfo"]] == [1, 2, 3]
        assert [c["ndr_type"] for c in h["channelinfo"]] == ["analog_in", "event", "mark"]

    def test_classic_son_aliases(self, fake_cli):
        """read_SOMSMR_header.m adds these so classic-SON callers keep working."""
        fi = read_SOMSMR_header(fake_cli)["fileinfo"]
        assert fi["usPerTime"] == 1
        assert fi["dTimeBase"] == fi["timebase"]
        assert fi["maxFTime"] == fi["max_time_ticks"]

    def test_channelinfo_selects_by_number(self, fake_cli):
        h = read_SOMSMR_header(fake_cli)
        assert sonpipe.channelinfo(h, 2)["kind_name"] == "EventRise"

    def test_channelinfo_unknown_channel_raises(self, fake_cli):
        h = read_SOMSMR_header(fake_cli)
        with pytest.raises(ValueError, match="not recorded"):
            sonpipe.channelinfo(h, 99)

    def test_channelinfo_empty_header_raises(self):
        with pytest.raises(ValueError, match="no channels"):
            sonpipe.channelinfo({"channelinfo": []}, 1)


class TestSampleInterval:
    def test_waveform_channel(self, fake_cli):
        si, total_samples, total_time = read_SOMSMR_sampleinterval(fake_cli, None, 1)
        assert si == pytest.approx(1.0 / SR)
        assert total_samples == N
        assert total_time == pytest.approx(0.5)

    def test_event_channel_reports_nan_not_none(self, fake_cli):
        """MATLAB's emptytonan: a null means not-applicable, not zero."""
        si, total_samples, total_time = read_SOMSMR_sampleinterval(fake_cli, None, 2)
        assert np.isnan(si) and np.isnan(total_samples)
        assert total_time == pytest.approx(1.0)


class TestWaveformRead:
    def test_full_read(self, fake_cli):
        data, total_samples, total_time, blockinfo, time = read_SOMSMR_datafile(
            fake_cli, None, 1, 0.0, float("inf")
        )
        assert data.shape == (N, 1)
        np.testing.assert_array_equal(data.ravel(), np.arange(N))
        assert total_samples == N
        assert blockinfo is None
        np.testing.assert_allclose(time.ravel(), np.arange(N) / SR)

    def test_time_window_maps_to_start_and_count(self, fake_cli):
        """t0/t1 become --start/--count; sample i has value i, so slips show."""
        data, _ts, _tt, _bi, time = read_SOMSMR_datafile(fake_cli, None, 1, 0.1, 0.2)
        np.testing.assert_array_equal(data.ravel(), np.arange(100, 201))
        np.testing.assert_allclose(time.ravel()[0], 0.1)
        np.testing.assert_allclose(time.ravel()[-1], 0.2)

    def test_negative_t0_is_clamped(self, fake_cli):
        data, _ts, _tt, _bi, _t = read_SOMSMR_datafile(fake_cli, None, 1, -5.0, 0.01)
        np.testing.assert_array_equal(data.ravel(), np.arange(0, 11))

    def test_empty_window_returns_empty(self, fake_cli):
        data, _ts, _tt, _bi, _t = read_SOMSMR_datafile(fake_cli, None, 1, 0.3, 0.2)
        assert data.size == 0

    def test_header_is_reused_when_supplied(self, fake_cli):
        h = read_SOMSMR_header(fake_cli)
        data, _ts, _tt, _bi, _t = read_SOMSMR_datafile(fake_cli, h, 1, 0.0, 0.01)
        assert data.size == 11


class TestEventAndMarkerRead:
    def test_event_times(self, fake_cli):
        data, _ts, _tt, _bi, time = read_SOMSMR_datafile(fake_cli, None, 2, 0.0, float("inf"))
        np.testing.assert_allclose(data.ravel(), [0.1, 0.2, 0.35, 0.5, 0.75])
        np.testing.assert_allclose(time.ravel(), data.ravel())

    def test_event_time_window(self, fake_cli):
        data, _ts, _tt, _bi, _t = read_SOMSMR_datafile(fake_cli, None, 2, 0.2, 0.5)
        np.testing.assert_allclose(data.ravel(), [0.2, 0.35, 0.5])

    def test_markers_return_times_and_codes(self, fake_cli):
        data, _ts, _tt, _bi, time = read_SOMSMR_datafile(fake_cli, None, 3, 0.0, float("inf"))
        np.testing.assert_allclose(time.ravel(), [0.15, 0.45, 0.85])
        np.testing.assert_allclose(data.ravel(), [11, 22, 33])


class TestFailureModes:
    """The sentinel exists because a sonpy abort() can look like success."""

    def test_missing_sentinel_is_a_crash(self, fake_cli, fault):
        fault("nosentinel")
        with pytest.raises(sonpipe.SonpipeError, match="did not report completion"):
            read_SOMSMR_datafile(fake_cli, None, 1, 0.0, float("inf"))

    def test_short_output_is_truncation(self, fake_cli, fault):
        fault("truncate")
        with pytest.raises(sonpipe.SonpipeError, match="truncated"):
            read_SOMSMR_datafile(fake_cli, None, 1, 0.0, float("inf"))

    def test_nonzero_exit_reports_the_command(self, fake_cli, fault):
        fault("fail")
        with pytest.raises(sonpipe.SonpipeError, match="simulated failure"):
            read_SOMSMR_header(fake_cli)

    def test_unparseable_json_is_reported(self, fake_cli, fault):
        fault("badjson")
        with pytest.raises(sonpipe.SonpipeError, match="did not return valid JSON"):
            read_SOMSMR_header(fake_cli)

    def test_unsupported_precision_rejected(self, fake_cli):
        with pytest.raises(ValueError, match="Unsupported precision"):
            sonpipe.invoke_binary(["read", fake_cli, "-c", "1"], "float128")


def test_fake_cli_is_faithful_to_the_real_argument_names():
    """Guard the stand-in against drift: it must accept the real CLI's flags."""
    argv = [
        sys.executable,
        str(FAKE),
        "read",
        "f.smrx",
        "-c",
        "1",
        "--start",
        "0",
        "--count",
        "5",
        "--t0",
        "0",
        "--t1",
        "1",
        "--json",
    ]
    completed = subprocess.run(argv, capture_output=True, env={**os.environ})
    assert completed.returncode == 0, completed.stderr.decode()
