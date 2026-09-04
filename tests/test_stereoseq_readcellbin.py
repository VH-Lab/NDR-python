"""readCellBin against the synthetic cellbin .h5ad files in cellbin_fixtures/.

Mirrors NDR-matlab's TestReadCellBin, fixture for fixture, so the two
readers are held to the same files rather than each to its own.

The fixtures are written by the REAL anndata library
(bscholl-genomics-python/cloudFriendly/make_cellbin_conformance_fixtures.py)
and read here by plain HDF5. That asymmetry is the point: NDR reads .h5ad
with raw HDF5 in both languages, so the fixtures must be genuine AnnData
output rather than our idea of the layout. If anndata changes how it lays
out categoricals or the obs index, these fail and say so.

Every fixture holds the same 5 cells with ragged real-vertex counts
(3, 4, 5, 8) and one cell with NONE, so a reader that assumes a fixed
count, or drops empty cells rather than keeping their row, shifts every
later cell's contour onto the wrong cell.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ndr.format.stereoseq import readCellBin

FIXTURES = pathlib.Path(__file__).resolve().parent / "cellbin_fixtures"

CELL_IDS = [f"{9000000000000 + i}" for i in range(5)]
REAL_VERTS = [3, 4, 5, 8, 0]


def cb(which: str) -> str:
    p = FIXTURES / f"cellbin_{which}.h5ad"
    assert p.is_file(), f"{p} missing; regenerate with make_cellbin_conformance_fixtures.py"
    return str(p)


def test_reads_cells_centroids_and_columns():
    cid, x, y, contours, obs, meta = readCellBin(cb("basic"))

    assert meta["nCells"] == 5
    assert cid == CELL_IDS
    assert meta["centroidSource"] == "obsm/spatial"
    assert x[0] == 10000.0 and y[0] == 20000.0
    assert set(obs) >= {"area", "dnbCount", "total_counts", "n_genes_by_counts"}
    assert list(obs["area"]) == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_cell_ids_stay_text():
    """13-digit identifiers lose precision as doubles and stop matching."""
    cid, *_ = readCellBin(cb("basic"))
    assert all(isinstance(c, str) for c in cid)
    assert cid[0] == "9000000000000"


def test_centroid_relative_is_detected():
    *_, meta = readCellBin(cb("basic"))
    assert meta["contourReference"] == "centroid"
    assert meta["contourReferenceSource"] == "detected"
    assert meta["relativeEvidence"]["ratio"] < meta["relativeEvidence"]["threshold"]


def test_absolute_is_detected():
    *_, meta = readCellBin(cb("absolute"))
    assert meta["contourReference"] == "absolute"
    assert meta["relativeEvidence"]["ratio"] > meta["relativeEvidence"]["threshold"]


def test_both_encodings_normalise_to_the_same_polygons():
    """The property that matters, and the one a threshold bug breaks.

    cellbin_basic and cellbin_absolute hold the SAME polygons, one stored
    centroid-relative and one absolute. Read into a common frame they must
    be identical. A wrong relative/absolute call puts every outline a
    chip-width from its cell without raising anything, so this is the test
    that would catch it.
    """
    for frame in ("centroid", "absolute"):
        _, _, _, rel, _, _ = readCellBin(cb("basic"), outputReference=frame)
        _, _, _, absol, _, _ = readCellBin(cb("absolute"), outputReference=frame)
        assert len(rel) == len(absol) == 5
        for a, b in zip(rel, absol):
            assert np.allclose(a, b), f"frame {frame} disagreed"


def test_padding_is_detected_and_reported():
    *_, meta = readCellBin(cb("basic"))
    assert meta["padValue"] == 32767.0
    assert 0.0 < meta["padFraction"] < 1.0


def test_empty_contour_keeps_its_row():
    """Dropping it would shift every later contour onto the wrong cell."""
    _, _, _, contours, _, meta = readCellBin(cb("basic"))
    assert [len(c) for c in contours] == REAL_VERTS
    assert contours[4].shape == (0, 2)
    assert meta["nEmptyContours"] == 1
    assert meta["raggedVertices"] is True


def test_centroids_from_obs_xy():
    _, x, y, _, _, meta = readCellBin(cb("obsxy"))
    assert meta["centroidSource"] == "obs x/y"
    assert x[0] == 10000.0 and y[0] == 20000.0


def test_missing_contours_reported_with_what_is_there():
    _, _, _, contours, _, meta = readCellBin(cb("nocontours"))
    assert meta["contoursPresent"] is False
    assert contours == []
    assert "spatial" in meta["obsmKeys"]


def test_label_columns_are_reported_not_chosen():
    """Both kinds appear, flagged, because the file does not say which."""
    *_, meta = readCellBin(cb("basic"))
    by_name = {d["name"]: d for d in meta["labelColumns"]}
    assert set(by_name) == {"leiden", "subclass_nn_column"}
    assert by_name["leiden"]["isUnsupervisedGuess"] is True
    assert by_name["subclass_nn_column"]["isUnsupervisedGuess"] is False
    assert by_name["subclass_nn_column"]["nCategories"] == 3


def test_reference_can_be_forced_and_says_so():
    _, _, _, forced, _, meta = readCellBin(cb("basic"), contourReference="absolute")
    assert meta["contourReferenceSource"] == "forced"
    _, _, _, auto, _, _ = readCellBin(cb("basic"))
    assert not np.allclose(forced[0], auto[0])


def test_pad_value_can_be_forced():
    *_, meta = readCellBin(cb("basic"), padValue=32767)
    assert meta["padValue"] == 32767.0


def test_probe_only_reports_without_returning_cells():
    cid, x, y, contours, obs, meta = readCellBin(cb("basic"), probeOnly=True)

    assert cid == [] and len(x) == 0 and contours == [] and obs == {}

    # The counts and every inference are still exact, which is the point:
    # a caller can show what is in a file before committing to reading it.
    assert meta["nCells"] == 5
    assert meta["contourReference"] == "centroid"
    assert meta["padValue"] == 32767.0
    assert [d["name"] for d in meta["labelColumns"]] == ["leiden", "subclass_nn_column"]


def test_obs_columns_can_be_selected():
    *_, obs, _ = readCellBin(cb("basic"), obsColumns=["area"])
    assert list(obs) == ["area"]


def test_unknown_obs_column_names_what_is_there():
    with pytest.raises(KeyError, match="nosuchcolumn"):
        readCellBin(cb("basic"), obsColumns=["nosuchcolumn"])


def test_not_an_h5ad_is_named():
    gef = pathlib.Path(__file__).resolve().parent / "gef_fixtures" / "gef_basic.gef"
    with pytest.raises(KeyError, match="no /obs group"):
        readCellBin(str(gef))
