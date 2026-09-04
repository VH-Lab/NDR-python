"""Read BGI/MGI Stereo-seq GEF files.

Port of ``+ndr/+format/+stereoseq/readGEF.m``.

A GEF is an HDF5 file holding one record per (pixel, gene) pair. This is
the raw-format layer: it returns arrays and knows nothing about NDI. The
first four are exactly what ``ndi.fun.doc.gene.makePyramid`` takes as its
first four arguments, which is the intended pairing, but nothing here
depends on it.

It does NOT follow the ``ndr.reader`` motif, and should not: a section is
not epoch-based data, and SAW carries far more parameters than that
interface has room for. It sits in ``ndr.format`` alongside the other
plain format readers, none of which implement that interface either.

Promoted from readStomicsGef.m in bscholl-genomics-python, which had been
reading real GEFs since August. The interface is unchanged from that
version because compareMatlabToPython had already validated it against
444 real tiles.

See also ``ndr.format.stereoseq.readCellBin``. NDR-matlab's
``docs/notes/stereoseq_formats.md`` describes both file layouts and the
SAW quirks these work around.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["readGEF"]

#: Layouts a Stereo-seq GEF is known to use. Probed in order.
_GEF_ROOTS = ("/geneExp/bin1", "/wholeExp/bin1")
_GEF_EXPR = ("expression", "cellBin")
_GEF_NAME = ("geneName", "geneID", "name", "gene")
_GEF_ID = ("geneID", "gene", "geneName", "name")
_GEF_OFFSET = ("offset", "offsets")
_GEF_COUNT = ("count", "counts")
_GEF_X = ("x", "X")
_GEF_Y = ("y", "Y")
_GEF_CNT = ("count", "MIDCount", "mid_count", "umi")

#: Records read per h5py call on the contiguous fast path.
_GEF_BLOCK = 20_000_000


def _gef_pick(available, candidates, what, where):
    """Return the first candidate present, or say what was looked for."""
    for c in candidates:
        if c in available:
            return c
    raise KeyError(
        f"No {what} field in {where}: looked for "
        f"{{{', '.join(candidates)}}}, found {{{', '.join(available)}}}."
    )


def _gef_str(values):
    """Decode a fixed-width HDF5 string column to a list of str."""
    out = []
    for v in values:
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        out.append(str(v).replace("\x00", "").strip())
    return out


def _gef_extent(handle, root, x, y):
    """Locate the bounding box, nearest-first from ``root`` outward.

    SAW writes minX/maxX/minY/maxY as attributes, but WHERE varies: on the
    bin group, on an ancestor, or on the file root. Probing only the root
    silently yields a 1x1 pyramid on files that put them deeper, and a
    fixture written with them at the root agrees with that bug rather than
    catching it.

    An attribute box that does not CONTAIN the data loses to the data,
    which cannot be wrong about its own extent.
    """
    need = ("minX", "minY", "maxX", "maxY")
    parts = root.strip("/").split("/")
    places = ["/" + "/".join(parts[:i]) for i in range(len(parts), 0, -1)]
    places.append("/")

    have_data = x is not None and len(x)
    if have_data:
        dmin = (int(x.min()), int(y.min()))
        dmax = (int(x.max()), int(y.max()))

    for p in places:
        if p != "/" and p not in handle:
            continue
        attrs = handle[p].attrs if p != "/" else handle.attrs
        if not all(k in attrs for k in need):
            continue
        box = tuple(int(attrs[k]) for k in need)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if not have_data:
            # Nothing to validate against, and a caller must be able to
            # tell that apart from a box the data agreed with.
            return box, f"attrs at {p} (unvalidated: no records read)"
        if box[0] <= dmin[0] and box[1] <= dmin[1] and box[2] >= dmax[0] and box[3] >= dmax[1]:
            return box, f"attrs at {p}"
    if have_data:
        return (dmin[0], dmin[1], dmax[0], dmax[1]), "data"
    return None, "unknown (no attributes found and no records read)"


def _gef_stat_totals(handle, gene_id):
    """SAW's own per-gene MIDcount from ``/stat/gene``, aligned to our rows.

    A third independent source for a number this project derives twice.
    Aligned by accession rather than by position: assuming the orders match
    would import our own ordering into the check meant to be independent
    of it.
    """
    if "/stat/gene" not in handle:
        return None, "absent (/stat/gene not in this file)"
    st = handle["/stat/gene"][...]
    names = st.dtype.names or ()
    id_f = _gef_pick(names, ("geneID", "gene", "geneName"), "gene id", "/stat/gene")
    c_f = _gef_pick(names, ("MIDcount", "MIDCount", "midcount", "count"), "count", "/stat/gene")
    ids = _gef_str(st[id_f])
    tot = np.asarray(st[c_f], dtype=np.int64)

    n = len(gene_id)
    if len(ids) >= n and ids[:n] == list(gene_id):
        return tot[:n], "row order identical to geneExp"
    lookup = dict(zip(ids, tot))
    out = np.full(n, np.nan)
    hit = 0
    for i, g in enumerate(gene_id):
        if g in lookup:
            out[i] = lookup[g]
            hit += 1
    return out, f"matched {hit}/{n} rows by geneID"


def readGEF(
    filename,
    probeOnly: bool = False,
    maxGenes: int = 0,
    countCeiling: int = 65535,
    root: str = "",
    verbose: bool = False,
):
    """Read a Stereo-seq GEF into the flat records :func:`makePyramid` takes.

    MATLAB equivalent: ``ndi.fun.doc.gene.readGEF``

    Returns one record per (pixel, gene) pair::

        x, y, gene_index, count, gene_id, gene_name, meta = readGEF(path)
        makePyramid(session, x, y, gene_index, count, gl, subjectID="s1")

    Args:
        filename: path to the .gef (an HDF5 file).
        probeOnly: read the gene table, extent and field names but NOT the
            expression records. ``x``, ``y``, ``gene_index`` and ``count``
            come back empty; ``meta["nRecords"]`` is still exact, because
            it is the sum of the gene table's own counts. A real section
            holds ~10^8 records and takes minutes, so a caller that only
            needs to SHOW what is in a file -- an ingest GUI listing genes,
            extent and chip before the user commits -- can ask without
            paying for the data.
        maxGenes: stop after this many genes; 0 reads all.
        countCeiling: counts saturate here rather than wrapping, and
            ``meta["nCountsClamped"]`` reports how many did. The default is
            what the tile format stores; it is an argument so a widened
            level does not need a new reader.
        root: force the HDF5 group, e.g. ``"/geneExp/bin1"``. Default ""
            probes the known layouts.
        verbose: print progress. Off by default because a library called
            from a GUI must not print.

    Returns:
        ``(x, y, gene_index, count, gene_id, gene_name, meta)``. ``x`` and
        ``y`` are int32 source coordinates, ``gene_index`` is a ZERO-BASED
        int32 row of ``gene_id``, ``count`` is uint16. ``meta`` carries
        ``root``, ``exprDataset``, ``fields``, ``nGenesInFile``,
        ``nGenes``, ``nRecords``, ``box``, ``boxSource``, ``resolutionNm``,
        ``chipSerial``, ``nCountsClamped``, ``statTotals``,
        ``statTotalsNote`` and ``readSeconds``.

    Raises:
        KeyError: if no (gene, expression) pair or a required field is
            absent. The message names what was looked for and what is
            there, because a GEF layout that drifts is the expected case.

    NOTHING ABOUT THE LAYOUT IS ASSUMED. Every path and field name is
    probed against a candidate list: records live under ``/geneExp/bin1``
    or ``/wholeExp/bin1``, the expression dataset is ``expression`` or
    ``cellBin``, coordinates are x/y or X/Y, and the count field has four
    spellings. GEF layouts drift between SAW versions.

    COUNTS ARE CLIPPED IN A WIDE TYPE. SAW writes uint8 counts at bin1;
    clipping in the narrow type saturates every value at 255 instead of at
    ``countCeiling``, silently, which is the trap that bit an earlier
    builder under numpy 2.
    """
    import time

    try:
        import h5py
    except ImportError as e:  # pragma: no cover - h5py ships with a core dep
        raise ImportError(
            "h5py is required to read a GEF. Install it with: pip install h5py"
            f"\n(Original error: {e})"
        ) from e

    t0 = time.time()
    roots = (root,) if root else _GEF_ROOTS

    with h5py.File(filename, "r") as f:
        # -- locate the (gene, expression) pair -------------------------
        chosen = expr_name = ""
        for r in roots:
            if r not in f or "gene" not in f[r]:
                continue
            for e in _GEF_EXPR:
                if e in f[r]:
                    chosen, expr_name = r, e
                    break
            if chosen:
                break
        if not chosen:
            raise KeyError(f"No (gene, expression) pair under {', '.join(roots)} in {filename}.")
        expr_ds = f[chosen][expr_name]
        gene_ds = f[chosen]["gene"]

        # -- gene table (small: tens of thousands of rows) --------------
        genes = gene_ds[...]
        gf = genes.dtype.names or ()
        name_field = _gef_pick(gf, _GEF_NAME, "gene name", f"{chosen}/gene")
        id_field = _gef_pick(gf, _GEF_ID, "gene id", f"{chosen}/gene")
        off_field = _gef_pick(gf, _GEF_OFFSET, "offset", f"{chosen}/gene")
        cnt_field = _gef_pick(gf, _GEF_COUNT, "count", f"{chosen}/gene")

        gene_name = _gef_str(genes[name_field])
        gene_id = _gef_str(genes[id_field])
        offsets = np.asarray(genes[off_field], dtype=np.int64)
        counts = np.asarray(genes[cnt_field], dtype=np.int64)

        n_genes_all = len(counts)
        n_genes = min(maxGenes, n_genes_all) if maxGenes > 0 else n_genes_all
        gene_name = gene_name[:n_genes]
        gene_id = gene_id[:n_genes]

        # -- expression field names, WITHOUT reading any records --------
        # The dtype is metadata, so probeOnly can report the layout it
        # would have used and a real read fails on a bad field name up
        # front rather than after minutes of reading.
        ef = expr_ds.dtype.names or ()
        x_f = _gef_pick(ef, _GEF_X, "x coordinate", expr_name)
        y_f = _gef_pick(ef, _GEF_Y, "y coordinate", expr_name)
        c_f = _gef_pick(ef, _GEF_CNT, "count", expr_name)

        n_records = int(counts[:n_genes].sum())
        meta: dict[str, Any] = {
            "root": chosen,
            "exprDataset": f"{chosen}/{expr_name}",
            "fields": {
                "x": x_f,
                "y": y_f,
                "count": c_f,
                "geneID": id_field,
                "geneName": name_field,
            },
            "nGenesInFile": n_genes_all,
            "nGenes": n_genes,
            "nRecords": n_records,
            "box": None,
            "boxSource": "",
            "resolutionNm": int(f.attrs["resolution"]) if "resolution" in f.attrs else 500,
            "chipSerial": _gef_str([f.attrs["sn"]])[0] if "sn" in f.attrs else "",
            "nCountsClamped": 0,
            "readSeconds": 0.0,
        }
        if verbose:
            print(f"[gef] {filename}")
            print(f"[gef] root={chosen} expr={expr_name}  genes={n_genes} of {n_genes_all}")
            print(f"[gef] expression fields: x={x_f} y={y_f} count={c_f}")

        if probeOnly:
            meta["box"], meta["boxSource"] = _gef_extent(f, chosen, None, None)
            meta["statTotals"], meta["statTotalsNote"] = _gef_stat_totals(f, gene_id)
            meta["readSeconds"] = time.time() - t0
            empty32 = np.zeros(0, np.int32)
            return empty32, empty32, empty32, np.zeros(0, np.uint16), gene_id, gene_name, meta

        # -- expression records -----------------------------------------
        x = np.zeros(n_records, np.int32)
        y = np.zeros(n_records, np.int32)
        count = np.zeros(n_records, np.uint16)
        n_clamped = 0

        # GEF lays records out grouped by gene, so when the offsets are
        # contiguous and ascending the gene of each record follows from the
        # counts alone and the file can be read in a few large blocks.
        # Worth checking rather than assuming: one read per gene is ~26,000
        # calls on a real file and the per-call overhead dominates.
        expected = np.concatenate(([0], np.cumsum(counts[:-1]))) if n_genes_all else offsets
        contiguous = np.array_equal(offsets, expected)
        if verbose:
            print(
                "[gef] gene offsets are contiguous and ascending; reading in blocks"
                if contiguous
                else "[gef] gene offsets are NOT contiguous; one read per gene (slower)"
            )

        if contiguous:
            gene_index = np.repeat(np.arange(n_genes, dtype=np.int32), counts[:n_genes])
            written = 0
            while written < n_records:
                n = min(_GEF_BLOCK, n_records - written)
                chunk = expr_ds[written : written + n]
                sl = slice(written, written + n)
                x[sl] = chunk[x_f].astype(np.int32)
                y[sl] = chunk[y_f].astype(np.int32)
                raw = chunk[c_f].astype(np.int64)  # WIDE type before clipping
                n_clamped += int((raw > countCeiling).sum())
                count[sl] = np.minimum(raw, countCeiling).astype(np.uint16)
                written += n
                if verbose:
                    print(f"[gef]   {written:,}/{n_records:,} records  {time.time()-t0:.0f}s")
        else:
            gene_index = np.zeros(n_records, np.int32)
            written = 0
            for i in range(n_genes):
                n = int(counts[i])
                if n == 0:
                    continue
                o = int(offsets[i])
                chunk = expr_ds[o : o + n]
                sl = slice(written, written + n)
                x[sl] = chunk[x_f].astype(np.int32)
                y[sl] = chunk[y_f].astype(np.int32)
                raw = chunk[c_f].astype(np.int64)
                n_clamped += int((raw > countCeiling).sum())
                count[sl] = np.minimum(raw, countCeiling).astype(np.uint16)
                gene_index[sl] = i
                written += n
                if verbose and i and i % 500 == 0:
                    print(f"[gef]   gene {i:,}/{n_genes:,}  {time.time()-t0:.0f}s")

        meta["nCountsClamped"] = n_clamped
        meta["box"], meta["boxSource"] = _gef_extent(f, chosen, x, y)
        meta["statTotals"], meta["statTotalsNote"] = _gef_stat_totals(f, gene_id)
        meta["readSeconds"] = time.time() - t0

        if verbose:
            b = meta["box"]
            print(
                f"[gef] extent {b[2]-b[0]+1} x {b[3]-b[1]+1} source units "
                f"(origin {b[0]},{b[1]}) from {meta['boxSource']}"
            )
            print(f"[gef] max count = {count.max()}, {n_clamped} clamped at {countCeiling}")

        return x, y, gene_index, count, gene_id, gene_name, meta
