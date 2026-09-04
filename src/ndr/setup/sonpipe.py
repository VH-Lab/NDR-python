"""Install the sonpipe CLI into the current Python environment.

Port of ``+ndr/+setup/sonpipe.m``. sonpipe wraps CED's ``sonpy`` binding and
is what the ``ndr.format.ced.sonpipe`` package drives out of process; without
it the CED reader raises ``SonpipeNotFoundError`` on the first call.

sonpipe is not on PyPI, and the vendor pins ``sonpy`` to specific interpreter
builds (CPython 3.14 on Linux at the time of writing). Both facts are why
this is a runtime step rather than a declared dependency, and why the same
install is also exposed as the ``[ced]`` optional extra in pyproject.toml --
that extra points at the same git URL this module installs, so which entry
point a caller uses does not change what lands.

Usage::

    python -m ndr.setup.sonpipe          # install
    python -m ndr.setup.sonpipe --check  # report what is or is not present

Or from Python::

    from ndr.setup.sonpipe import install
    install()
"""

from __future__ import annotations

import argparse
import subprocess
import sys

SONPIPE_GIT_URL = "sonpipe @ git+https://github.com/VH-Lab/sonpipe.git"


class SonpipeInstallError(RuntimeError):
    """Raised when the pip install of sonpipe fails."""


def install(*, upgrade: bool = False) -> None:
    """Install sonpipe into the current interpreter via pip.

    Raises ``SonpipeInstallError`` if pip returns non-zero. The error message
    carries pip's own stderr so the failure -- typically an incompatible
    interpreter for CED's ``sonpy`` -- is visible without re-running the
    command by hand.
    """
    argv = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        argv.append("--upgrade")
    argv.append(SONPIPE_GIT_URL)

    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SonpipeInstallError(
            "pip install of sonpipe failed (exit "
            f"{completed.returncode}). stderr:\n{completed.stderr.strip()}"
        )


def _check() -> int:
    """Report whether the sonpipe CLI is reachable through the NDR bridge."""
    from ndr.format.ced import sonpipe as bridge

    bridge.reset_cache()
    try:
        argv = bridge.executable()
    except bridge.SonpipeNotFoundError as exc:
        print(f"sonpipe: NOT FOUND\n  {exc}", file=sys.stderr)
        return 1
    print("sonpipe:", " ".join(argv))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ndr.setup.sonpipe",
        description="Install the sonpipe CLI so the NDR CED reader can talk to CED's sonpy.",
    )
    parser.add_argument("--upgrade", action="store_true", help="Pass --upgrade to pip install.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Skip installing; only report whether sonpipe is reachable.",
    )
    args = parser.parse_args(argv)

    if args.check:
        return _check()

    try:
        install(upgrade=args.upgrade)
    except SonpipeInstallError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return _check()


if __name__ == "__main__":
    raise SystemExit(main())
