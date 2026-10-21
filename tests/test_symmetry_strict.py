"""Positive controls for the symmetry tests' strict mode (VH-Lab/NDR-python#34).

``tests/symmetry/conftest.py`` turns any skipped symmetry test into a failure
when ``NDR_SYMMETRY_CHECK_STRICT`` is set, which is what keeps a comparison
that never ran from reporting green in ``test-symmetry.yml``. That guard is
only worth anything if it demonstrably fires, so these tests run one real
symmetry read test against an EMPTY artifact base, once without the flag (it
must skip) and once with it (it must fail). They live here rather than under
tests/symmetry/ because the default run ignores that directory, and a guard
that is only exercised by the job it guards is not exercised at all.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# Any symmetry read test will do; this one needs only the checked-in example
# videos and PyAV, both present wherever the full suite runs.
READ_TEST = "tests/symmetry/read_artifacts/reader/test_read_video.py"

pytest.importorskip("av")


def _run(tmp_path, strict):
    env = dict(os.environ)
    # The symmetry artifacts live under tempfile.gettempdir(); an empty TMPDIR
    # guarantees the other language's artifacts are absent.
    env["TMPDIR"] = str(tmp_path)
    env.pop("NDR_SYMMETRY_CHECK_STRICT", None)
    if strict:
        env["NDR_SYMMETRY_CHECK_STRICT"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfs", READ_TEST],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )


def test_missing_artifacts_skip_outside_strict_mode(tmp_path):
    result = _run(tmp_path, strict=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout and "failed" not in result.stdout, result.stdout


def test_missing_artifacts_fail_in_strict_mode(tmp_path):
    result = _run(tmp_path, strict=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "was SKIPPED while NDR_SYMMETRY_CHECK_STRICT is set" in result.stdout, result.stdout
    assert " skipped" not in result.stdout.splitlines()[-1], result.stdout
