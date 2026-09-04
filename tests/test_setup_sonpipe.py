"""Cover ndr.setup.sonpipe without actually pip-installing anything.

The real install lands in the ced-integration CI job (which runs `pip install
-e ".[dev,ced]"`); those tests would be a network round-trip and depend on
whichever interpreter CED's sonpy happens to ship a wheel for. Here we mock
subprocess and exercise the flow.
"""

from __future__ import annotations

import subprocess
import sys
from unittest import mock

import pytest

from ndr.setup import sonpipe as setup_sonpipe


class TestInstall:
    def test_invokes_pip_with_the_git_url(self, monkeypatch):
        seen: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        setup_sonpipe.install()
        assert seen["argv"] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            setup_sonpipe.SONPIPE_GIT_URL,
        ]

    def test_upgrade_adds_the_flag(self, monkeypatch):
        seen: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        setup_sonpipe.install(upgrade=True)
        assert "--upgrade" in seen["argv"]

    def test_nonzero_pip_raises_with_stderr(self, monkeypatch):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="pip broke\n")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(setup_sonpipe.SonpipeInstallError, match="pip broke"):
            setup_sonpipe.install()


class TestCli:
    def test_check_reports_when_sonpipe_is_reachable(self, monkeypatch, capsys):
        with mock.patch("ndr.format.ced.sonpipe.executable", return_value=["/opt/sonpipe"]):
            with mock.patch("ndr.format.ced.sonpipe.reset_cache"):
                rc = setup_sonpipe.main(["--check"])
        out = capsys.readouterr().out
        assert rc == 0 and "sonpipe: /opt/sonpipe" in out

    def test_check_reports_when_sonpipe_is_missing(self, monkeypatch, capsys):
        from ndr.format.ced import sonpipe as bridge

        def raiser():
            raise bridge.SonpipeNotFoundError("nope")

        with mock.patch("ndr.format.ced.sonpipe.executable", side_effect=raiser):
            with mock.patch("ndr.format.ced.sonpipe.reset_cache"):
                rc = setup_sonpipe.main(["--check"])
        err = capsys.readouterr().err
        assert rc == 1 and "NOT FOUND" in err

    def test_install_failure_returns_nonzero(self, monkeypatch, capsys):
        def raise_install(*a, **kw):
            raise setup_sonpipe.SonpipeInstallError("pip exited 2")

        monkeypatch.setattr(setup_sonpipe, "install", raise_install)
        rc = setup_sonpipe.main([])
        err = capsys.readouterr().err
        assert rc == 1 and "pip exited 2" in err
