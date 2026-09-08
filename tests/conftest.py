"""Shared pytest configuration for the top-level test suite.

Its one job today: mark the bridge tests that need an NDR-matlab checkout, so
the test-matrix jobs can exclude them instead of skipping them.

WHY THAT MATTERS. The matrix jobs check out only NDR-python, so every
MATLAB-dependent bridge test skipped there -- 79 skips per job, three jobs. In
a repo whose whole bridge guard exists because "a check that could not run
must not report the same result as a check that ran and passed", a standing
pile of skips is the exact smell we are trying to remove: it trains a reader
to scroll past skips, and the next one to appear -- a real one -- goes
unnoticed. The `bridge` job, which does check out NDR-matlab, still runs
every one of them.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

#: Applied automatically; see :func:`pytest_collection_modifyitems`.
NEEDS_MATLAB = "needs_matlab"

#: The helper a test calls when it cannot run without the MATLAB tree.
MATLAB_GATE = "require_matlab_root"

#: Set by the bridge job once both repos are checked out. Duplicated from
#: test_matlab_bridge_completeness rather than imported, because a conftest
#: that imports a test module at collection time is a circular-import waiting
#: to happen. test_matlab_bridge_conventions asserts the two agree.
STRICT_ENV_VAR = "NDR_BRIDGE_CHECK_STRICT"

#: Test modules whose skips are not allowed to pass unnoticed in CI.
BRIDGE_TEST_PREFIX = "test_matlab_bridge_"


def pytest_collection_modifyitems(config, items):
    """Mark every test that reaches for the NDR-matlab checkout.

    DERIVED, NOT HAND-APPLIED. The marker is read off the test's own source:
    anything calling ``require_matlab_root`` is marked. A hand-maintained list
    of decorators would drift the first time somebody added a MATLAB-dependent
    test and forgot one -- and the symptom (one quiet skip in the matrix) is
    precisely what this arrangement exists to prevent.

    KNOWN LIMIT: a test that reaches the gate only through a helper of its own
    is not detected, because only the test function's source is scanned. Every
    such test today calls the gate directly (13 tests, 13 call sites). If that
    ever stops being true, mark the test explicitly with
    ``@pytest.mark.needs_matlab``; the symptom until then is benign and
    visible -- a nonzero skip count in the matrix jobs.
    """
    for item in items:
        function = getattr(item, "function", None)
        if function is None:
            continue
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):  # pragma: no cover - source always available here
            continue
        if MATLAB_GATE in source:
            item.add_marker(getattr(pytest.mark, NEEDS_MATLAB))


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Under strict mode, a skipped bridge test is a FAILED bridge test.

    WHY THIS EXISTS. The bridge guard's founding rule is that a check which
    could not run must not report the same result as a check that ran and
    passed. Three separate skip paths are gated on ``NDR_BRIDGE_CHECK_STRICT``
    individually -- a missing MATLAB tree, a shallow clone, an absent
    merge-base -- but gating them one at a time is a list somebody has to
    remember to extend. The drift self-tests already carry three ungated
    ``pytest.skip`` calls of their own: if NDR-matlab's history ever made
    their fixture conditions true, the positive controls proving the drift
    rule discriminates would stop running, silently, with CI green. Those
    skips are reasonable locally -- a test that cannot build its fixture
    should not fail a developer's laptop -- but in CI the fixtures are
    expected to exist, so a skip there is a hole, not a courtesy.

    So this is a backstop rather than another entry on the list: under strict
    mode ANY skip in a bridge test module fails, including ones nobody
    thought to gate. Derived, not enumerated -- the same reasoning as the
    ``needs_matlab`` marker above.

    Outside strict mode nothing changes, so a developer without an NDR-matlab
    checkout still sees ordinary skips.
    """
    report = yield

    if not os.environ.get(STRICT_ENV_VAR, "").strip():
        return report
    if not report.skipped:
        return report
    if hasattr(report, "wasxfail"):  # an xfail is a recorded expectation, not a gap
        return report
    if not Path(str(item.fspath)).name.startswith(BRIDGE_TEST_PREFIX):
        return report

    reason = ""
    if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        reason = report.longrepr[2]
    report.outcome = "failed"
    report.longrepr = (
        f"{item.nodeid} was SKIPPED while {STRICT_ENV_VAR} is set.\n\n"
        f"    reason: {reason}\n\n"
        "In CI every bridge check is expected to be able to run, so a skip "
        "here is a check reporting the same result as one that passed -- the "
        "exact failure this guard exists to prevent. Either make the "
        "condition hold in CI, or gate the skip on "
        f"{STRICT_ENV_VAR} yourself with a message saying why it could not "
        "run. Do not delete the assertion to make this green."
    )
    return report
