"""Enumerate tiles for a SmartSPIM channel.

Port of ``+ndr/+format/+smartspim/listTiles.m``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import validate_call

from ndr.format.smartspim.readAcquisitionMetadata import readAcquisitionMetadata
from ndr.format.smartspim.readStitcherXml import readStitcherXml

__all__ = ["listTiles"]


@validate_call
def listTiles(rootDir: str, channelName: str) -> list[dict[str, Any]]:
    """Enumerate tiles for one SmartSPIM channel.

    Returns one dict per tile in the given channel, built from the
    channel's ``xml_import.xml`` (mandatory) and ``xml_merging.xml``
    (optional -- only present after stitching). Fields:

    - ``id`` (str): the tile's DIR_NAME from the XML (e.g.
      ``"389960/389960_445310"``). Always uses forward-slash separators.
    - ``row`` (float): tile grid row (0-indexed, from XML)
    - ``col`` (float): tile grid column (0-indexed, from XML)
    - ``numSlices`` (float): z-slice count for this tile (from
      metadata.json when available; NaN otherwise)
    - ``nominalPixelOffset`` (list[float] length 3): [d, v, h] pixel offset
      from ``xml_import.xml`` (absolute tile offset before stitch refinement)
    - ``nominalUmOffset`` (list[float] length 3): [z, y, x] in micrometers
    - ``stitchedPixelOffset`` (list[float] length 3 or None): [d, v, h]
      from ``xml_merging.xml``, or None if merging XML is not present or
      does not carry a matching stack
    - ``stitchedUmOffset`` (list[float] length 3 or None): [z, y, x] in
      micrometers, or None

    The tile order mirrors ``xml_import.xml`` -- Stack elements as they
    appear in the file. A missing ``xml_merging.xml`` is not an error --
    stitched offsets simply come back as None.
    """
    if not rootDir:
        raise ValueError("rootDir must be a non-empty string.")
    if not channelName:
        raise ValueError("channelName must be a non-empty string.")
    channel_dir = Path(rootDir) / channelName
    if not channel_dir.is_dir():
        raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

    import_path = channel_dir / "xml_import.xml"
    if not import_path.is_file():
        raise FileNotFoundError(f"xml_import.xml not found in {channel_dir}")
    import_xml = readStitcherXml(str(import_path))

    merging_path = channel_dir / "xml_merging.xml"
    merging_by_dir: dict[str, dict[str, Any]] = {}
    voxel_dims_merging = import_xml["voxelDims"]
    if merging_path.is_file():
        merging_xml = readStitcherXml(str(merging_path))
        voxel_dims_merging = merging_xml["voxelDims"]
        for s in merging_xml["stacks"]:
            merging_by_dir[s["dirName"]] = s

    num_slices_by_dir: dict[str, float] = {}
    meta_path = Path(rootDir) / "metadata.json"
    if meta_path.is_file():
        try:
            meta = readAcquisitionMetadata(rootDir)
            for t in meta["tiles"]:
                if t["channelName"] != channelName:
                    continue
                dn = f"{t['X']}/{t['X']}_{t['Y']}"
                num_slices_by_dir[dn] = t["numImages"]
        except (OSError, ValueError):
            num_slices_by_dir = {}

    v_i = [
        import_xml["voxelDims"]["D"],
        import_xml["voxelDims"]["V"],
        import_xml["voxelDims"]["H"],
    ]
    v_m = [voxel_dims_merging["D"], voxel_dims_merging["V"], voxel_dims_merging["H"]]

    tiles: list[dict[str, Any]] = []
    for s in import_xml["stacks"]:
        key = s["dirName"]
        n = num_slices_by_dir.get(key, float("nan"))
        nom_pix = [s["absD"], s["absV"], s["absH"]]
        nom_um = [p * v for p, v in zip(nom_pix, v_i)]
        m = merging_by_dir.get(key)
        if m is not None:
            st_pix = [m["absD"], m["absV"], m["absH"]]
            st_um = [p * v for p, v in zip(st_pix, v_m)]
        else:
            st_pix = None
            st_um = None
        tiles.append(
            {
                "id": s["dirName"],
                "row": s["row"],
                "col": s["col"],
                "numSlices": float(n),
                "nominalPixelOffset": nom_pix,
                "nominalUmOffset": nom_um,
                "stitchedPixelOffset": st_pix,
                "stitchedUmOffset": st_um,
            }
        )
    return tiles
