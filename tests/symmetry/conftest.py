"""Shared fixtures and configuration for NDR symmetry tests."""

import os
import tempfile
from pathlib import Path

import pytest

from ndr.fun.ndrpath import ndrpath

#: Set by .github/workflows/test-symmetry.yml. Under it, a skipped symmetry
#: test FAILS; see :func:`pytest_runtest_makereport`.
STRICT_ENV_VAR = "NDR_SYMMETRY_CHECK_STRICT"

# Base directory where all symmetry artifacts live:
#   <tempdir>/NDR/symmetryTest/<sourceType>/<namespace>/<class>/<test>/
SYMMETRY_BASE = Path(tempfile.gettempdir()) / "NDR" / "symmetryTest"
PYTHON_ARTIFACTS = SYMMETRY_BASE / "pythonArtifacts"
MATLAB_ARTIFACTS = SYMMETRY_BASE / "matlabArtifacts"

SOURCE_TYPES = ["matlabArtifacts", "pythonArtifacts"]

# The checked-in example data, resolved the same way MATLAB does it
# (`fullfile(ndr.fun.ndrpath(), 'example_data')`). Deriving this from
# ndrpath() rather than from __file__ keeps it correct regardless of where
# the tests are run from or whether the package is installed in place.
EXAMPLE_DATA = ndrpath() / "example_data"


def json_safe(value):
    """Convert a Python value into something both ports encode identically.

    MATLAB's ``jsonencode(..., 'ConvertInfAndNaN', true)`` writes NaN and Inf
    as ``null``; Python's ``json.dumps`` would emit bare ``NaN``/``Infinity``,
    which is not valid JSON and would not round-trip through MATLAB's
    ``jsondecode``. Numpy scalars and arrays are converted to plain Python so
    the encodings match element for element.

    2-D arrays become nested row lists, which is what MATLAB's ``jsonencode``
    produces for a matrix, so image planes compare correctly despite MATLAB
    being column-major and numpy row-major.
    """
    import math

    import numpy as np

    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Under strict mode, a skipped symmetry test is a FAILED symmetry test.

    A read test skips when the other language produced no artifacts, and a
    test of either kind skips when an example file or an optional dependency
    is missing. Each of those is reasonable on a developer's machine, where
    only one port may be installed. In CI, every comparison is expected to
    run, so a skip there means the comparison did not happen while the job
    still reported green. That is what hid the video symmetry tests on
    VH-Lab/NDR-python#33's first run (VH-Lab/NDR-python#34).

    So, as in tests/conftest.py for the bridge tests, this is a backstop and
    not a list: under strict mode ANY skip in a test under tests/symmetry/
    fails, including one nobody thought to gate. Cases that cannot run in a
    given CI stage by design (the pythonArtifacts read cases in Stage 2,
    before Python has made them) are deselected by that stage, not skipped.
    """
    report = yield

    if not os.environ.get(STRICT_ENV_VAR, "").strip():
        return report
    if not report.skipped or hasattr(report, "wasxfail"):
        return report
    if Path(__file__).parent not in Path(str(item.fspath)).parents:
        return report

    reason = ""
    if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        reason = report.longrepr[2]
    report.outcome = "failed"
    report.longrepr = (
        f"{item.nodeid} was SKIPPED while {STRICT_ENV_VAR} is set.\n\n"
        f"    reason: {reason}\n\n"
        "In CI every symmetry comparison is expected to run, so a skip here "
        "reports the same result as a comparison that ran and matched. If the "
        "other language's artifacts are missing, check that its half of the "
        "test exists and that the workflow checked out the right branch of "
        "the other repository (see the job summary). Do not delete the "
        "comparison to make this green."
    )
    return report
