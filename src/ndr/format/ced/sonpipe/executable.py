"""Locate, set, or query the sonpipe command-line tool.

Port of +ndr/+format/+ced/+sonpipe/executable.m.

sonpipe is a separate process on purpose. CED publishes ``sonpy`` only as
prebuilt binaries, and on Linux and macOS no release installs on CPython 3.10
through 3.13 at all; on Apple Silicon the macOS build is x86_64-only and cannot
be imported by a native arm64 interpreter on any version. Running the reader
out of process lets NDR-python talk to whatever interpreter and architecture
sonpy needs without constraining its own.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

_CACHED: list[str] | None = None


class SonpipeNotFoundError(RuntimeError):
    """Raised when the sonpipe CLI cannot be located."""


def _default_install_candidates() -> list[list[str]]:
    """Absolute paths where sonpipe's install.sh / install.ps1 place the tool."""
    paths: list[Path] = []
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        paths.append(Path(home) / ".local" / "bin" / "sonpipe")
        paths.append(Path(home) / ".local" / "share" / "sonpipe" / "venv" / "bin" / "sonpipe")
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        paths.append(Path(localappdata) / "sonpipe" / "venv" / "Scripts" / "sonpipe.exe")
    return [[str(p)] for p in paths if p.is_file()]


def _works(argv: list[str]) -> bool:
    try:
        completed = subprocess.run(
            [*argv, "--version"], capture_output=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def executable(newvalue: str | list[str] | None = None) -> list[str]:
    """Return the argv prefix that invokes the sonpipe CLI.

    Pass ``newvalue`` to set and cache it explicitly, for an install that is not
    on PATH::

        executable("/opt/venv/bin/sonpipe")
        executable([sys.executable, "-m", "sonpipe"])

    Lookup order when nothing is cached mirrors executable.m: the ``SONPIPE``
    environment variable, ``sonpipe`` on PATH, the default install locations,
    then ``-m sonpipe`` under this interpreter and the usual python names. Each
    candidate is verified by running ``--version``.

    Unlike the MATLAB port this keeps argv as a list and never builds a shell
    string, so a path or filename containing spaces or quotes needs no escaping.
    """
    global _CACHED

    if newvalue is not None:
        _CACHED = shlex.split(newvalue) if isinstance(newvalue, str) else list(newvalue)
        return _CACHED

    if _CACHED is not None:
        return _CACHED

    candidates: list[list[str]] = []
    env = os.environ.get("SONPIPE")
    if env:
        candidates.append(shlex.split(env))
    candidates.append(["sonpipe"])
    candidates.extend(_default_install_candidates())
    candidates.append([sys.executable, "-m", "sonpipe"])
    candidates.append(["python3", "-m", "sonpipe"])
    candidates.append(["python", "-m", "sonpipe"])

    for candidate in candidates:
        if _works(candidate):
            _CACHED = candidate
            return _CACHED

    raise SonpipeNotFoundError(
        "Could not locate the sonpipe command-line tool, which NDR uses to read "
        "CED Spike2 files. Install it from https://github.com/VH-Lab/sonpipe, or "
        "point NDR at an existing install with "
        "ndr.format.ced.sonpipe.executable(PATH) or the SONPIPE environment "
        "variable."
    )


def reset_cache() -> None:
    """Forget the cached executable. Intended for tests."""
    global _CACHED
    _CACHED = None
