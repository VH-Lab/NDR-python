"""Run the sonpipe CLI and capture its output.

Port of +ndr/+format/+ced/+sonpipe/private/invoke_text.m and invoke_binary.m.

The MATLAB versions route stdout through a temporary file because MATLAB's
system() mangles binary captured into a char array, and they clear
LD_LIBRARY_PATH so a child Python does not inherit MATLAB's bundled libraries.
Neither applies here: subprocess returns clean bytes, and Python does not
inject its own libraries into children. What does carry over is the completion
sentinel check, which is not a MATLAB workaround -- see check_sentinel below.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

import numpy as np

from ndr.format.ced.sonpipe.executable import executable

# "sonpipe: wrote 12345 samples (double) for channel 3"
# "sonpipe: wrote 0 event times (double) for channel 5"
_WROTE = re.compile(r"wrote\s+(\d+)\s+(?:samples|event times)")

_DTYPES = {
    "double": "<f8",
    "single": "<f4",
    "float32": "<f4",
    "float64": "<f8",
    "int16": "<i2",
    "int32": "<i4",
}


class SonpipeError(RuntimeError):
    """The sonpipe CLI failed, crashed, or returned truncated output."""


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    argv = [*executable(), *args]
    try:
        return subprocess.run(argv, capture_output=True, check=False)
    except OSError as err:  # pragma: no cover - depends on the local install
        raise SonpipeError(f"Could not run sonpipe: {' '.join(argv)}") from err


def _fail(args: list[str], completed: subprocess.CompletedProcess[bytes]) -> None:
    raise SonpipeError(
        f"sonpipe failed (status {completed.returncode}) for command:\n"
        f"  sonpipe {' '.join(args)}\n"
        f"{completed.stderr.decode('utf-8', 'replace')}"
    )


def invoke_text(args: list[str]) -> str:
    """Run sonpipe and return stdout as text."""
    completed = _run(args)
    if completed.returncode != 0:
        _fail(args, completed)
    return completed.stdout.decode("utf-8", "replace")


def invoke_json(args: list[str]) -> Any:
    """Run sonpipe and parse stdout as JSON."""
    text = invoke_text(args)
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        raise SonpipeError(
            f"sonpipe did not return valid JSON for command:\n  sonpipe {' '.join(args)}\n"
            f"{text[:2000]}"
        ) from err


def invoke_binary(args: list[str], precision: str = "double") -> np.ndarray:
    """Run sonpipe and read its raw little-endian stdout as a 1-D array.

    Verifies the CLI's completion sentinel. sonpy is a compiled library that on
    some files fails an assertion and calls abort() (SIGABRT) mid-stream rather
    than raising. The exit status alone does not catch that: on Apple Silicon
    the CLI runs behind an ``arch -x86_64`` wrapper, and the wrapper does not
    reliably propagate a signal death, so a hard crash can look like success and
    return short data. A completed read prints "sonpipe: wrote N ..." to stderr
    as its final act, and N must match what we captured.
    """
    if precision not in _DTYPES:
        raise ValueError(f"Unsupported precision {precision!r}; expected one of {sorted(_DTYPES)}")

    completed = _run(args)
    stderr = completed.stderr.decode("utf-8", "replace")
    if completed.returncode != 0:
        _fail(args, completed)

    data = np.frombuffer(completed.stdout, dtype=np.dtype(_DTYPES[precision]))

    match = _WROTE.search(stderr)
    if match is None:
        raise SonpipeError(
            f"sonpipe did not report completion for command:\n  sonpipe {' '.join(args)}\n"
            "The reader process appears to have crashed before finishing (a sonpy "
            "assertion/abort reads as SIGABRT). Captured messages:\n"
            f"{stderr}"
        )
    expected = int(match.group(1))
    if expected != data.size:
        raise SonpipeError(
            f"sonpipe reported {expected} value(s) but {data.size} were captured for "
            f"command:\n  sonpipe {' '.join(args)}\n"
            "The output is truncated -- the reader likely crashed mid-stream. "
            f"Captured messages:\n{stderr}"
        )
    return np.array(data, dtype=np.float64 if precision in ("double", "float64") else data.dtype)
