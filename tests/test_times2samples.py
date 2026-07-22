"""Regression tests for MATLAB round-half-away-from-zero in times2samples.

numpy/Python round halves to even (banker's rounding); MATLAB rounds halves away
from zero. On exact half-sample boundaries the two disagree, which surfaces as
intermittent MATLAB<->Python symmetry failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from ndr.time.fun.times2samples import matlab_round, times2samples


@pytest.mark.parametrize(
    "value,expected",
    [
        (2.5, 3.0),
        (3.5, 4.0),
        (0.5, 1.0),
        (-0.5, -1.0),
        (-2.5, -3.0),
        (1.4, 1.0),
        (1.6, 2.0),
    ],
)
def test_matlab_round_half_away_from_zero(value, expected):
    assert float(matlab_round(value)) == expected


def test_times2samples_half_boundary():
    # 0.0025 * 1000 = 2.5 -> MATLAB round = 3, +1 base = 4 (numpy banker's -> 3).
    assert float(times2samples(0.0025, [0, 10], 1000)) == 4.0
    # 0.0035 * 1000 = 3.5 -> 4, +1 = 5.
    assert float(times2samples(0.0035, [0, 10], 1000)) == 5.0


def test_times2samples_array_unchanged():
    s = times2samples(np.array([0.0, 0.1, 0.2]), (0.0, 1.0), 10.0)
    assert np.allclose(s, [1.0, 2.0, 3.0])


def test_times2samples_infinities():
    s = times2samples(np.array([-np.inf, np.inf]), (0.0, 1.0), 10.0)
    assert s[0] == 1
    assert s[1] == 11
