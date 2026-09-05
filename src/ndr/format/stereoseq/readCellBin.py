"""Read SAW cellbin .h5ad cell segmentation files.

Port of ``+ndr/+format/+stereoseq/readCellBin.m``.

This is the raw-format layer: it returns arrays and knows nothing about
NDI, though its outputs are what ``ndi.fun.doc.gene.makeCells`` takes.

READ WITH PLAIN HDF5, NOT WITH anndata. An .h5ad is an HDF5 file with
AnnData conventions layered on top, and those conventions are simple
enough to read directly: ``/obs`` carries an ``_index`` attribute naming
the identifier dataset, numeric columns are datasets, and categorical
columns are GROUPS holding ``categories`` and ``codes``. Reading it
directly is what lets MATLAB and Python do the same thing here; an
anndata dependency would leave the two ports on different footings, which
is what the bridge exists to prevent.

See also ``ndr.format.stereoseq.readGEF``. NDR-matlab's
``docs/notes/stereoseq_formats.md`` describes both file layouts and the
SAW quirks these work around.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["readCellBin"]

#: Vertices count as centroid-relative when their median magnitude falls
#: below this fraction of the centroid scale. See readCellBin's docstring.
_RELATIVE_THRESHOLD = 0.05

#: A sentinel must fill at least this share of the slots to be believed.
_PAD_DOMINANCE = 0.25

_XY_PAIRS = (("x", "y"), ("spatial_x", "spatial_y"), ("X", "Y"))


def _text(values) -> list[str]:
    out = []
    for v in np.atleast_1d(values):
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        out.append(str(v).replace("\x00", "").strip())
    return out


def _index_name(obs) -> str:
    nm = obs.attrs.get("_index", "_index")
    if isinstance(nm, bytes):
        nm = nm.decode("utf-8", "replace")
    return str(nm)


def _classify_obs(obs, index_name):
    """Split /obs into numeric columns and categorical label columns."""
    import h5py

    numeric, labels = [], []
    for name, node in obs.items():
        if name in (index_name, "_index"):
            continue
        if isinstance(node, h5py.Group):
            if "categories" in node and "codes" in node:
                labels.append(
                    {
                        "name": name,
                        "nCategories": int(node["categories"].shape[0]),
                        "isUnsupervisedGuess": _unsupervised(name),
                    }
                )
            continue
        numeric.append(name)
    return sorted(numeric), sorted(labels, key=lambda d: d["name"])


def _categorical(group, name: str, n_cells: int) -> list[str]:
    """Decode an AnnData categorical /obs column into one string per cell.

    The column is a GROUP, not a dataset: ``categories`` holds the distinct
    strings and ``codes`` a ZERO-BASED index into them, one per cell. pandas
    writes -1 for a value it has no category for, and that is a meaningful
    state rather than a defect -- a cell the labeling never assigned -- so
    it becomes ``""``, which is what ``makeCellTypeLabels`` documents an
    empty label to mean.

    Both bounds are CHECKED rather than trusted. A code array of the wrong
    length, or one reaching past the categories, would otherwise put a real
    category name on the wrong cell, and nothing downstream could tell:
    every label would still be a legal label. Note that numpy's negative
    indexing makes -1 silently address the LAST category, so the empty
    string here is a deliberate branch, not a fallback.
    """
    cats = _text(group["categories"][...])
    codes = np.asarray(group["codes"][...]).ravel()
    if codes.size != n_cells:
        raise ValueError(
            f"/obs/{name}/codes has {codes.size} entries " f"but the file has {n_cells} cells."
        )
    if codes.size and codes.max() > len(cats) - 1:
        raise ValueError(
            f"/obs/{name}/codes reaches category {int(codes.max())} "
            f"but only {len(cats)} are defined."
        )
    if codes.size and codes.min() < -1:
        raise ValueError(
            f"/obs/{name}/codes holds {int(codes.min())}; "
            f"the only negative code pandas writes is -1."
        )
    return ["" if c < 0 else cats[c] for c in codes]


def _unsupervised(name: str) -> bool:
    """Guess, from the NAME alone, whether a labeling is unsupervised.

    An unsupervised clustering carries no biological identity; a
    transferred atlas call is an inference about each cell. The file does
    not distinguish them, and treating one as the other is a scientific
    error rather than a display bug. Marked as a guess for that reason.
    """
    low = name.lower()
    return "leiden" in low or "louvain" in low or "snn_res" in low or low.startswith("cluster")


def _centroid_source(f, numeric):
    if "obsm" in f and "spatial" in f["obsm"]:
        return "obsm/spatial", True
    for xk, yk in _XY_PAIRS:
        if xk in numeric and yk in numeric:
            return f"obs {xk}/{yk}", False
    raise KeyError("No centroids: need obsm/spatial or an obs x/y pair.")


def _centroids(f, source, has_spatial):
    if has_spatial:
        s = np.asarray(f["obsm/spatial"][...], dtype=float)
        return s[:, 0].copy(), s[:, 1].copy()
    xk, yk = source.replace("obs ", "").split("/")
    return (
        np.asarray(f[f"obs/{xk}"][...], dtype=float),
        np.asarray(f[f"obs/{yk}"][...], dtype=float),
    )


def _pad(cb, requested):
    """The sentinel filling the unused vertex slots, and how much it fills."""
    flat = cb.reshape(-1, 2)
    if requested != "auto":
        pad = float("nan") if requested is None else float(requested)
    else:
        col = flat[:, 0]
        col = col[np.isfinite(col)]
        if col.size == 0:
            pad = float("nan")
        else:
            vals, counts = np.unique(col, return_counts=True)
            # Believe a sentinel only if it dominates. A file with no
            # padding must not have its most common real coordinate
            # mistaken for one.
            pad = (
                float(vals[counts.argmax()])
                if counts.max() > _PAD_DOMINANCE * col.size
                else float("nan")
            )
    frac = 0.0 if np.isnan(pad) else float(np.all(flat == pad, axis=1).mean())
    return pad, frac


def _reference(cb, pad, x, y, requested, threshold):
    flat = cb.reshape(-1, 2)
    if not np.isnan(pad):
        flat = flat[~np.all(flat == pad, axis=1)]
    real_med = float(np.nanmedian(np.abs(flat))) if flat.size else float("nan")
    scale = max(float(np.abs(x).max()), float(np.abs(y).max()), 1.0)
    ratio = real_med / scale
    evidence = {
        "realVertexAbsMedian": real_med,
        "centroidScale": scale,
        "ratio": ratio,
        "threshold": threshold,
    }
    if requested == "auto":
        ref = "absolute" if np.isnan(ratio) else ("centroid" if ratio < threshold else "absolute")
        return ref, "detected", evidence
    return requested, "forced", evidence


def _polygons(cb, pad, ref, x, y, output_ref):
    relative = ref == "centroid"
    polys, vpc = [], []
    for i in range(cb.shape[0]):
        v = np.asarray(cb[i], dtype=float)
        keep = np.isfinite(v).all(axis=1)
        if not np.isnan(pad):
            keep &= ~np.all(v == pad, axis=1)
        if relative:
            # When vertices are centroid-relative, (0,0) IS the centroid
            # and never a real boundary point, so it is padding whatever
            # the sentinel search concluded.
            keep &= ~np.all(v == 0, axis=1)
        v = v[keep]
        if v.shape[0] >= 2:  # drop repeats, including a closing vertex
            v = v[~np.all(v == np.roll(v, 1, axis=0), axis=1)]
        if v.shape[0] < 3:
            polys.append(np.empty((0, 2)))
            vpc.append(0)
            continue
        if relative and output_ref == "absolute":
            v = v + np.array([x[i], y[i]])
        elif not relative and output_ref == "centroid":
            v = v - np.array([x[i], y[i]])
        polys.append(v)
        vpc.append(v.shape[0])
    return polys, np.asarray(vpc)


def readCellBin(
    filename,
    probeOnly: bool = False,
    contourReference: str = "auto",
    padValue: Any = "auto",
    outputReference: str = "centroid",
    obsColumns: list[str] | None = None,
):
    """Read a SAW cellbin .h5ad.

    MATLAB equivalent: ``ndr.format.stereoseq.readCellBin``

    Returns one entry per cell: an identifier, a centroid, an optional
    boundary polygon, and whatever per-cell measurements the file carries::

        cid, x, y, contours, obs, meta = readCellBin(path)
        makeCells(session, cid, x, y, pyr, contours=contours,
                  contourReference=meta["contourReference"])

    Args:
        filename: path to the cellbin .h5ad.
        probeOnly: report what the file contains -- cell count, available
            columns, label candidates, and the contour inferences -- without
            returning identifiers, centroids or contours. For a caller that
            must show a file before committing to it.
        contourReference: ``"auto"``, ``"centroid"`` or ``"absolute"``. See
            THE FILE DOES NOT SAY below.
        padValue: ``"auto"``, a numeric sentinel, or None for none.
        outputReference: frame the contours come back in, ``"centroid"``
            (offsets from each cell's centroid) or ``"absolute"``.
            Centroid-relative is the default because that is what
            ``spatialGeneExpressionCells`` stores and what int16 vertices
            can hold.
        obsColumns: per-cell columns to return; None returns every numeric
            one. A CATEGORICAL column may be named here too, and comes back
            as a list of str of its decoded category names, one per cell --
            that is how a labeling gets out of the file and into
            ``ndi.fun.doc_gene.makeCellTypeLabels``. It is not in the
            default because the default is measurements, and a labeling is
            a claim about each cell rather than a measurement of it: which
            labeling to believe is the caller's decision (see
            ``meta["labelColumns"]``), so it must be asked for by name.

    Returns:
        ``(cell_id, x, y, contours, obs, meta)``. ``cell_id`` is a list of
        str -- TEXT, never numbers, because these are commonly 13-14 digit
        identifiers that lose precision as a double and then no longer
        match the file they came from. ``contours`` is a list of (V, 2)
        arrays with padding removed; a cell with no usable boundary gets a
        (0, 2) and ITS ROW IS KEPT, because dropping it would shift every
        later cell's contour onto the wrong cell, silently.

    THE FILE DOES NOT SAY whether boundary vertices are stored relative to
    each cell's centroid or in absolute source coordinates, nor what value
    pads the unused vertex slots. Both are INFERRED, and both are recorded
    as explicit fields on the NDI document precisely so nothing downstream
    has to guess again.

    So this reports rather than deciding quietly:
    ``meta["contourReference"]`` is the answer,
    ``meta["contourReferenceSource"]`` says whether it was detected or
    forced, and ``meta["relativeEvidence"]`` carries the numbers behind
    it. Getting it wrong is silent and total: treat absolute vertices as
    relative and every outline lands a chip-width from its cell, with no
    error raised.

    The rule: vertices are centroid-relative when the median magnitude of
    the REAL (non-padding) vertices is under 5% of the centroid scale,
    where the scale is ``max(max|x|, max|y|, 1)``. Padding is excluded
    first because a sentinel of 32767 would otherwise dominate the median
    and mask small relative offsets. Two earlier implementations of this
    test disagreed -- one used ``median(|centroid|)`` with a 10%
    threshold, the other ``max`` with 5% -- and agreed only by luck on the
    data they had run on; consolidating them is half the reason this
    function exists. The scale uses max rather than a median because it
    describes the frame's extent rather than where the tissue happens to
    sit.

    LABEL COLUMNS ARE REPORTED, NOT CHOSEN. A cellbin routinely carries
    several: transferred atlas calls and unsupervised clusterings side by
    side. They are not interchangeable, and the file does not say which is
    which, so ``meta["labelColumns"]`` lists them with an
    ``isUnsupervisedGuess`` flag derived from the column NAME alone.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - h5py is in the formats extra
        raise ImportError(
            "h5py is required to read a cellbin .h5ad. "
            "Install it with: pip install 'ndr[formats]'"
            f"\n(Original error: {e})"
        ) from e

    with h5py.File(filename, "r") as f:
        if "obs" not in f:
            raise KeyError(f"{filename} has no /obs group; this does not look like an .h5ad.")
        obs_group = f["obs"]
        index_name = _index_name(obs_group)
        cell_id = _text(obs_group[index_name][...])
        n_cells = len(cell_id)

        numeric, labels = _classify_obs(obs_group, index_name)
        source, has_spatial = _centroid_source(f, numeric)

        meta: dict[str, Any] = {
            "nCells": n_cells,
            "nGenes": int(f["var"][_index_name(f["var"])].shape[0]) if "var" in f else None,
            "centroidSource": source,
            "obsColumns": numeric + [d["name"] for d in labels],
            "labelColumns": labels,
            "contoursPresent": "obsm" in f and "cell_border" in f["obsm"],
            "contourShape": None,
            "padValue": float("nan"),
            "padFraction": float("nan"),
            "contourReference": "",
            "contourReferenceSource": "",
            "relativeEvidence": {
                "realVertexAbsMedian": float("nan"),
                "centroidScale": float("nan"),
                "ratio": float("nan"),
                "threshold": _RELATIVE_THRESHOLD,
            },
            "verticesPerCell": (float("nan"),) * 3,
            "raggedVertices": False,
            "nEmptyContours": float("nan"),
        }
        if not meta["contoursPresent"]:
            meta["obsmKeys"] = sorted(f["obsm"].keys()) if "obsm" in f else []

        # Centroids are needed even under probeOnly: the relative/absolute
        # inference is a comparison AGAINST them, so reporting it without
        # them would be reporting a guess with no evidence.
        x = y = np.zeros(0)
        if meta["contoursPresent"] or not probeOnly:
            x, y = _centroids(f, source, has_spatial)

        contours: list[np.ndarray] = []
        if meta["contoursPresent"]:
            cb = np.asarray(f["obsm/cell_border"][...], dtype=float)
            if cb.ndim != 3 or cb.shape[2] != 2:
                raise ValueError(
                    f"obsm/cell_border has shape {cb.shape}; expected " "(n_cells, vertices, 2)."
                )
            meta["contourShape"] = tuple(int(v) for v in cb.shape)

            pad, frac = _pad(cb, padValue)
            meta["padValue"], meta["padFraction"] = pad, frac

            ref, ref_src, evid = _reference(cb, pad, x, y, contourReference, _RELATIVE_THRESHOLD)
            meta["contourReference"] = ref
            meta["contourReferenceSource"] = ref_src
            meta["relativeEvidence"] = evid

            contours, vpc = _polygons(cb, pad, ref, x, y, outputReference)
            meta["verticesPerCell"] = (
                int(vpc.min()),
                float(np.median(vpc)),
                int(vpc.max()),
            )
            meta["raggedVertices"] = bool(vpc.min() != vpc.max())
            meta["nEmptyContours"] = int((vpc == 0).sum())

        if probeOnly:
            return [], np.zeros(0), np.zeros(0), [], {}, meta

        want = obsColumns if obsColumns else numeric
        label_names = [d["name"] for d in labels]
        obs: dict[str, Any] = {}
        for c in want:
            if c in numeric:
                obs[c] = np.asarray(obs_group[c][...])
            elif c in label_names:
                obs[c] = _categorical(obs_group[c], c, meta["nCells"])
            else:
                raise KeyError(f"No /obs column {c!r}; available: {numeric + label_names}")

        return cell_id, x, y, contours, obs, meta
