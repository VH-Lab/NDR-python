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

import pytest

#: Applied automatically; see :func:`pytest_collection_modifyitems`.
NEEDS_MATLAB = "needs_matlab"

#: The helper a test calls when it cannot run without the MATLAB tree.
MATLAB_GATE = "require_matlab_root"


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
