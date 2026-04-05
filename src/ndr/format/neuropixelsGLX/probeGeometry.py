"""Extract probe geometry from a SpikeGLX .meta file.

Port of +ndr/+format/+neuropixelsGLX/probeGeometry.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.neuropixelsGLX.header import header
from ndr.format.neuropixelsGLX.readmeta import readmeta


def _find_meta_field(meta: dict[str, str], target: str) -> str:
    """Find a field in the meta dict by partial name match.

    Handles the fact that readmeta transforms '~' prefixed field names
    (e.g. ~snsGeomMap becomes _snsGeomMap).
    """
    for key in meta:
        if target.lower() in key.lower():
            return meta[key]
    return ""


def _parse_snsGeomMap(
    geommap_str: str, n_neural: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the snsGeomMap string for electrode positions.

    Format: (header)(shank:x:y:used)(shank:x:y:used)...
    Header: (probePN,nShanks,shankSep,shankWidth) - skipped.
    Each channel entry has colon-separated integers: shank:x:y:used.

    Returns
    -------
    x, y, shank : ndarray
        Electrode x positions, y (depth) positions, and shank IDs (1-based).
    """
    tokens = re.findall(r"\(([^)]+)\)", geommap_str)

    if len(tokens) < 1 + n_neural:
        raise ValueError(
            f"snsGeomMap has {len(tokens)} entries but expected at least "
            f"{1 + n_neural} (header + {n_neural} neural)."
        )

    x = np.zeros(n_neural)
    y = np.zeros(n_neural)
    shank = np.zeros(n_neural, dtype=int)

    # Skip the first token (header), parse channel entries
    for i in range(n_neural):
        parts = tokens[i + 1].split(":")
        if len(parts) < 4:
            raise ValueError(
                f"Could not parse snsGeomMap entry {i}: \"{tokens[i + 1]}\"."
            )
        shank[i] = int(parts[0]) + 1  # Convert 0-based to 1-based
        x[i] = float(parts[1])
        y[i] = float(parts[2])

    return x, y, shank


def _compute_geometry_from_imro(
    imroTbl_str: str, probe_type: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute electrode positions from imroTbl.

    Supported probe types:
        0          - Neuropixels 1.0 (staggered 2-column, 20 um pitch)
        21, 2003, 2004 - Neuropixels 2.0 single shank (2-column, 15 um pitch)
        24, 2013, 2014 - Neuropixels 2.0 four shank (2-column, 15 um pitch)

    Returns
    -------
    x, y, shank : ndarray
        Electrode x positions, y (depth) positions, and shank IDs (1-based).
    """
    tokens = re.findall(r"\(([^)]+)\)", imroTbl_str)
    if len(tokens) < 2:
        raise ValueError("Could not parse imroTbl: too few parenthesized groups.")

    n_chans = len(tokens) - 1  # First token is the header
    x = np.zeros(n_chans)
    y = np.zeros(n_chans)
    shank = np.ones(n_chans, dtype=int)

    if probe_type == 0:
        # Neuropixels 1.0
        # Entry format: (chanIdx bank refIdx apGain lfGain apHiPass)
        # Electrode ID = chanIdx + bank * 384
        # Staggered 2-column layout, 20 um vertical pitch
        for i in range(n_chans):
            vals = [int(v) for v in tokens[i + 1].replace(",", " ").split()]
            elec_id = vals[0] + vals[1] * 384
            row = elec_id // 2
            col = elec_id % 2
            y[i] = row * 20
            if row % 2 == 0:
                x[i] = 27 + col * 32
            else:
                x[i] = 11 + col * 32

    elif probe_type in (21, 2003, 2004):
        # Neuropixels 2.0 single shank
        # Entry format: (chanIdx bankMask refIdx elecInd)
        # 2-column layout, 15 um vertical pitch
        for i in range(n_chans):
            vals = [int(v) for v in tokens[i + 1].replace(",", " ").split()]
            elec_ind = vals[3]
            row = elec_ind // 2
            col = elec_ind % 2
            y[i] = row * 15
            x[i] = 27 + col * 32

    elif probe_type in (24, 2013, 2014):
        # Neuropixels 2.0 four shank
        # Entry format: (chanIdx shankIdx bankMask refIdx elecInd)
        # 2-column layout per shank, 15 um pitch, 250 um shank spacing
        for i in range(n_chans):
            vals = [int(v) for v in tokens[i + 1].replace(",", " ").split()]
            shank_idx = vals[1]
            elec_ind = vals[4]
            row = elec_ind // 2
            col = elec_ind % 2
            y[i] = row * 15
            x[i] = shank_idx * 250 + 27 + col * 32
            shank[i] = shank_idx + 1

    else:
        raise ValueError(
            f"Probe type {probe_type} is not supported for geometry computation "
            f"from imroTbl. Use a SpikeGLX version that writes ~snsGeomMap "
            f"to the .meta file."
        )

    return x, y, shank


def _probe_type_info(probe_type: int) -> tuple[str, float]:
    """Return probe model name and contact size for a probe type.

    Returns
    -------
    model_name : str
    contact_size : float
        Contact width/height in micrometers.
    """
    if probe_type == 0:
        return "Neuropixels 1.0", 12.0
    elif probe_type in (21, 2003, 2004):
        return "Neuropixels 2.0 Single Shank", 12.0
    elif probe_type in (24, 2013, 2014):
        return "Neuropixels 2.0 Four Shank", 12.0
    elif probe_type in (1100, 1110):
        return "Neuropixels Ultra", 6.0
    else:
        return f"Neuropixels (type {probe_type})", 12.0


def probeGeometry(metafilename: str | Path) -> dict[str, Any]:
    """Extract probe geometry from a SpikeGLX .meta file.

    Reads a SpikeGLX .meta file and returns a dictionary whose fields match
    the NDI probe_geometry document type. Electrode positions are taken
    from the ~snsGeomMap field if present (SpikeGLX >= 20230202), or
    computed from the ~imroTbl field and probe type for older files.

    Parameters
    ----------
    metafilename : str or Path
        Full path to a SpikeGLX .meta file.

    Returns
    -------
    dict
        Probe geometry with fields:
        - site_locations_leftright : ndarray, [N x 1] left-right positions in um
        - site_locations_frontback : ndarray, [N x 1] zeros
        - site_locations_depth : ndarray, [N x 1] depth positions in um
        - probe_model : str
        - manufacturer : str ('IMEC')
        - shank_id : ndarray, [N x 1] shank IDs (1-based)
        - contact_shape : str ('square')
        - contact_shape_width : ndarray, [N x 1] in um
        - contact_shape_height : ndarray, [N x 1] in um
        - contact_shape_radius : ndarray, [N x 1] zeros
        - ndim : int (2)
        - unit : str ('um')
        - has_planar_contour : int (0)
        - contour_x : ndarray (empty)
        - contour_y : ndarray (empty)
    """
    metafilename = Path(metafilename)
    meta = readmeta(metafilename)
    info = header(metafilename)
    n_neural = info["n_neural_chans"]

    # Try snsGeomMap first (available in newer SpikeGLX)
    geom_str = _find_meta_field(meta, "snsGeomMap")

    if geom_str:
        x, y, shank = _parse_snsGeomMap(geom_str, n_neural)
    else:
        # Fall back to computing geometry from imroTbl + probe type
        imro_str = _find_meta_field(meta, "imroTbl")
        if not imro_str:
            raise ValueError("Meta file has neither snsGeomMap nor imroTbl.")
        probe_type_num = int(info["probe_type"]) if info["probe_type"] else 0
        x, y, shank = _compute_geometry_from_imro(imro_str, probe_type_num)

    # Get probe model name and contact size
    probe_type_num = int(info["probe_type"]) if info["probe_type"] else 0
    model_name, contact_size = _probe_type_info(probe_type_num)

    return {
        "site_locations_leftright": x.reshape(-1),
        "site_locations_frontback": np.zeros(n_neural),
        "site_locations_depth": y.reshape(-1),
        "probe_model": model_name,
        "manufacturer": "IMEC",
        "shank_id": shank.reshape(-1),
        "contact_shape": "square",
        "contact_shape_width": np.full(n_neural, contact_size),
        "contact_shape_height": np.full(n_neural, contact_size),
        "contact_shape_radius": np.zeros(n_neural),
        "ndim": 2,
        "unit": "um",
        "has_planar_contour": 0,
        "contour_x": np.array([]),
        "contour_y": np.array([]),
    }
