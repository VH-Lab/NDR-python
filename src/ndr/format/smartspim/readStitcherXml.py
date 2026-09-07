"""Parse a TeraStitcher-format XML file used by SmartSPIM.

Port of ``+ndr/+format/+smartspim/readStitcherXml.m``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from pydantic import validate_call

__all__ = ["readStitcherXml"]

# The vendor XML DOCTYPE names a "TeraStitcher.DTD" that is not shipped.
# Strip it (and drop it as an XXE attack surface) before parsing.
_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>")


def _get_attr(el: ET.Element, name: str) -> str:
    v = el.get(name)
    return v if v is not None else ""


def _parse_triple(root: ET.Element, tag_name: str, xml_path: Path) -> dict[str, float]:
    el = root.find(tag_name)
    if el is None:
        raise ValueError(f"{xml_path}: missing <{tag_name}> element")
    out: dict[str, float] = {}
    for a in ("V", "H", "D"):
        v = _get_attr(el, a)
        if not v:
            raise ValueError(f'{xml_path}: <{tag_name}> missing attribute "{a}"')
        out[a] = float(v)
    return out


def _parse_origin_optional(root: ET.Element) -> dict[str, float] | None:
    el = root.find("origin")
    if el is None:
        return None
    out: dict[str, float] = {}
    for a in ("V", "H", "D"):
        v = _get_attr(el, a)
        out[a] = float(v) if v else 0.0
    return out


def _parse_dimensions(root: ET.Element, xml_path: Path) -> dict[str, float]:
    el = root.find("dimensions")
    if el is None:
        raise ValueError(f"{xml_path}: missing <dimensions> element")
    out: dict[str, float] = {}
    attrs = [
        ("stack_rows", "stackRows"),
        ("stack_columns", "stackColumns"),
        ("stack_slices", "stackSlices"),
    ]
    for src, dst in attrs:
        v = _get_attr(el, src)
        if not v:
            raise ValueError(f'{xml_path}: <dimensions> missing attribute "{src}"')
        out[dst] = float(v)
    return out


def _parse_stacks(root: ET.Element, xml_path: Path) -> list[dict[str, Any]]:
    stacks_el = root.find("STACKS")
    if stacks_el is None:
        raise ValueError(f"{xml_path}: missing <STACKS> element")
    nodes = stacks_el.findall("Stack")
    if not nodes:
        raise ValueError(f"{xml_path}: no <Stack> elements found")
    required = ("ROW", "COL", "ABS_V", "ABS_H", "ABS_D", "DIR_NAME")
    stacks: list[dict[str, Any]] = []
    for i, el in enumerate(nodes):
        for a in required:
            if not _get_attr(el, a):
                raise ValueError(f'{xml_path}: <Stack> #{i + 1} missing attribute "{a}"')
        stacks.append(
            {
                "row": float(_get_attr(el, "ROW")),
                "col": float(_get_attr(el, "COL")),
                "absV": float(_get_attr(el, "ABS_V")),
                "absH": float(_get_attr(el, "ABS_H")),
                "absD": float(_get_attr(el, "ABS_D")),
                "dirName": _get_attr(el, "DIR_NAME"),
                "zRanges": _get_attr(el, "Z_RANGES"),
            }
        )
    return stacks


@validate_call
def readStitcherXml(xmlPath: str) -> dict[str, Any]:
    """Parse a TeraStitcher XML file into a dict.

    Returned fields:

    - ``voxelDims``: dict with keys V, H, D (micrometers)
    - ``dimensions``: dict with keys stackRows, stackColumns, stackSlices
    - ``origin``: dict with keys V, H, D, or None if absent
    - ``stacks``: list of dicts (one per ``<Stack>``) with row, col,
      absV, absH, absD, dirName, zRanges

    Defensive against the vendor XML pointing at an unreachable DTD: the
    DOCTYPE line is stripped before parsing, and no external entity
    resolution occurs (ElementTree does none by default).
    """
    if not xmlPath:
        raise ValueError("xmlPath must be a non-empty string.")
    p = Path(xmlPath)
    if not p.is_file():
        raise FileNotFoundError(f"XML file not found: {xmlPath}")

    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as err:
        raise OSError(f"Failed to read {xmlPath}: {err}") from err

    stripped = _DOCTYPE_RE.sub("", raw, count=1)

    try:
        root = ET.fromstring(stripped)
    except ET.ParseError as err:
        raise ValueError(f"Failed to parse {xmlPath}: {err}") from err

    return {
        "voxelDims": _parse_triple(root, "voxel_dims", p),
        "dimensions": _parse_dimensions(root, p),
        "origin": _parse_origin_optional(root),
        "stacks": _parse_stacks(root, p),
    }
