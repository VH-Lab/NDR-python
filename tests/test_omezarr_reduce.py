"""Tests for :func:`ndr.format.omezarr.reduce`.

Python mirror of ``tools/tests/+ndr/+unittest/+format/+omezarr/TestOMEZarrReduce.m``.
Self-contained numeric tests: mean vs max, dtype behavior (max preserves,
mean promotes to double), per-axis factor vector, trailing partial block
kept, arity and reduction validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ndr.format.omezarr.reduce import reduce


def test_mean_scalar_factor():
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    r = reduce(a, 2, "mean")
    assert r.shape == (1, 1)
    assert r[0, 0] == pytest.approx(2.5, abs=1e-12)


def test_max_scalar_factor():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    r = reduce(a, 2, "max")
    assert r.shape == (1, 1)
    assert r[0, 0] == np.uint16(4)
    assert r.dtype == np.uint16


def test_mean_promotes_to_double():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    r = reduce(a, 2, "mean")
    assert r.dtype == np.float64


def test_trailing_partial_block_kept():
    # An axis of length 5 with factor 2 must produce 3 samples (blocks of
    # 2, 2, 1). Matches how the lab's NGFF writers size coarser levels.
    a = np.arange(1, 6)
    r = reduce(a, 2, "max")
    assert r.shape == (3,)
    np.testing.assert_array_equal(r, np.array([2, 4, 5]))


def test_per_axis_factor_vector():
    # 4x6 with factors [2, 3] -> 2x2 output. Mean case.
    a = np.arange(1, 25).reshape(4, 6)
    r = reduce(a, [2, 3], "mean")
    assert r.shape == (2, 2)


def test_wrong_factor_arity_errors():
    a = np.zeros((3, 4))
    with pytest.raises(ValueError):
        reduce(a, [2, 2, 2], "mean")


def test_factor_must_be_positive_integer():
    a = np.zeros((4, 4))
    with pytest.raises(ValueError):
        reduce(a, 0, "mean")


def test_reduction_must_be_member():
    a = np.zeros((4, 4))
    with pytest.raises(ValueError):
        reduce(a, 2, "median")


def test_factor_one_passes_through():
    # Factor 1 on every axis should reproduce the input (mean returns
    # double, max preserves dtype).
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    r = reduce(a, 1, "max")
    np.testing.assert_array_equal(r, a)
    assert r.dtype == np.uint16
