"""Tests for :func:`ndr.format.omezarr.probe`.

Python mirror of ``tools/tests/+ndr/+unittest/+format/+omezarr/TestOMEZarrProbe.m``.
Exercises the cheap ``ok/nPyramids/pyramidNames/axesOrder/level0*`` fields
against the shared programmatic OME-Zarr fixture, plus the not-ok paths
(missing dir, empty dir).
"""

from __future__ import annotations

import pytest

from ndr.format.omezarr.probe import probe
from tests._omezarr_fixture import make_example_fixture


@pytest.fixture
def fixture_dir(tmp_path):
    return make_example_fixture(tmp_path, with_chunks=False)[0]


def test_ok_on_fixture(fixture_dir):
    info = probe(str(fixture_dir))
    assert info.ok is True


def test_not_ok_on_missing_dir(tmp_path):
    info = probe(str(tmp_path / "does-not-exist"))
    assert info.ok is False
    assert info.n_pyramids == 0
    assert info.pyramid_names == []


def test_not_ok_on_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    info = probe(str(empty))
    assert info.ok is False


def test_pyramid_count_and_names(fixture_dir):
    info = probe(str(fixture_dir))
    assert info.n_pyramids == 2
    assert "mean" in info.pyramid_names
    assert "max" in info.pyramid_names


def test_first_pyramid_metadata(fixture_dir):
    info = probe(str(fixture_dir))
    assert info.axes_order == "czyx"
    assert len(info.axes_units) == 4
    assert len(info.level0_shape) == 4
    assert len(info.level0_chunks) == 4
    assert len(info.level0_scale) == 4
    assert info.dtype != ""


def test_n_levels_per_pyramid(fixture_dir):
    info = probe(str(fixture_dir))
    assert len(info.n_levels_per_pyramid) == 2
    assert all(n >= 1 for n in info.n_levels_per_pyramid)
