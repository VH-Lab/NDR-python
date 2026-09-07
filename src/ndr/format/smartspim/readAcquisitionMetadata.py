"""Parse SmartSPIM acquisition metadata.json (and optional sequence.json).

Port of ``+ndr/+format/+smartspim/readAcquisitionMetadata.m``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import validate_call

__all__ = ["readAcquisitionMetadata"]


def _get_field_safe(s: Any, name: str, default: Any) -> Any:
    if isinstance(s, dict) and name in s:
        return s[name]
    return default


def _find_field_by_prefix(s: Any, prefix_lower: str) -> Any:
    """Case-insensitive prefix match against dict keys.

    Handy for JSON keys mangled between writers (e.g. "laser power" vs
    "laserPower"). Returns None if no match.
    """
    if not isinstance(s, dict):
        return None
    for key in s:
        if isinstance(key, str) and key[: len(prefix_lower)].lower() == prefix_lower:
            return s[key]
    return None


def _numeric_from_any(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return float("nan")
    return float("nan")


def _coerce_char(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return str(v)
    return ""


def _char_field(s: Any, name: str, default: str) -> str:
    if not isinstance(s, dict) or name not in s:
        return default
    c = _coerce_char(s[name])
    if c == "":
        return default
    return c


def _numeric_field(s: Any, name: str, default: float) -> float:
    if not isinstance(s, dict) or name not in s:
        return default
    n = _numeric_from_any(s[name])
    if math.isnan(n):
        return default
    return n


def _coerce_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _parse_laser_power(raw: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    if raw is None:
        return out
    for e in _coerce_list(raw):
        if not isinstance(e, dict):
            continue
        out.append(
            {
                "wavelength": _numeric_from_any(_get_field_safe(e, "wavelength", None)),
                "leftPct": _numeric_from_any(_find_field_by_prefix(e, "left")),
                "rightPct": _numeric_from_any(_find_field_by_prefix(e, "right")),
            }
        )
    return out


def _parse_tiles(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if raw is None:
        return out
    for t in _coerce_list(raw):
        if not isinstance(t, dict):
            continue
        laser = _numeric_from_any(_get_field_safe(t, "Laser", None))
        filt = _coerce_char(_get_field_safe(t, "Filter", ""))
        fch = _numeric_from_any(_get_field_safe(t, "FilterChannel", None))
        ch_name = ""
        if not math.isnan(laser) and filt and not math.isnan(fch):
            ch_name = f"Ex_{laser:g}_Em_{filt}_Ch{fch:g}"
        out.append(
            {
                "X": _coerce_char(_get_field_safe(t, "X", "")),
                "Y": _coerce_char(_get_field_safe(t, "Y", "")),
                "Z": _coerce_char(_get_field_safe(t, "Z", "")),
                "laser": laser,
                "side": _numeric_from_any(_get_field_safe(t, "Side", None)),
                "exposure": _numeric_from_any(_get_field_safe(t, "Exposure", None)),
                "filter": filt,
                "filterChannel": fch,
                "numImages": _numeric_from_any(_get_field_safe(t, "NumImages", None)),
                "channelName": ch_name,
            }
        )
    return out


def _parse_sequence(seq_struct: Any, seq_path: Path) -> dict[str, Any]:
    steps = _get_field_safe(seq_struct, "steps", None)
    if not steps:
        raise ValueError(f'sequence.json is missing "steps": {seq_path}')
    step_list = _coerce_list(steps)
    step = step_list[0]

    z = _get_field_safe(step, "z", None)
    if not z or not isinstance(z, dict):
        raise ValueError(f'sequence.json step is missing "z": {seq_path}')

    imaging_raw = _find_field_by_prefix(step, "imaging")
    imaging_steps: list[dict[str, Any]] = []
    if imaging_raw is not None:
        for ci in _coerce_list(imaging_raw):
            imaging_steps.append(
                {
                    "laser": _numeric_from_any(_get_field_safe(ci, "laser", None)),
                    "filter": _coerce_char(_get_field_safe(ci, "filter", "")),
                }
            )

    laser_raw = _find_field_by_prefix(step, "laser")
    tile_boundary = _find_field_by_prefix(step, "tile")
    if not tile_boundary:
        tile_boundary = None

    focus_raw = _find_field_by_prefix(step, "focus")
    focus_points: list[Any] = [] if not focus_raw else _coerce_list(focus_raw)

    return {
        "objective": _char_field(seq_struct, "objective", ""),
        "immersion": _char_field(seq_struct, "immersion", ""),
        "name": _char_field(step, "name", ""),
        "zStepUm": _numeric_from_any(_find_field_by_prefix(z, "step")),
        "zTopUm": _numeric_from_any(_find_field_by_prefix(z, "top")),
        "zBottomUm": _numeric_from_any(_find_field_by_prefix(z, "bottom")),
        "imagingSteps": imaging_steps,
        "laserPower": _parse_laser_power(laser_raw),
        "tileBoundary": tile_boundary,
        "focusPoints": focus_points,
    }


@validate_call
def readAcquisitionMetadata(rootDir: str) -> dict[str, Any]:
    """Return the parsed contents of ``rootDir/metadata.json`` (and sequence.json).

    Errors if ``metadata.json`` is missing, unparseable, or missing
    required fields (``sample_metadata``, ``um_per_pix``, ``z_step_um``).
    Errors if ``sequence.json`` is present but malformed.

    See the MATLAB header for the full field-by-field description; the
    Python port preserves every field name verbatim.
    """
    if not rootDir:
        raise ValueError("rootDir must be a non-empty string.")
    root = Path(rootDir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {rootDir}")
    meta_path = root / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json not found in {rootDir}")

    try:
        with open(meta_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except ValueError as err:
        raise ValueError(f"Failed to parse {meta_path} as JSON: {err}") from err

    if "sample_metadata" not in raw:
        raise ValueError(f'metadata.json is missing "sample_metadata": {meta_path}')
    sm = raw["sample_metadata"]

    for req in ("um_per_pix", "z_step_um"):
        if req not in sm:
            raise ValueError(f'sample_metadata is missing required field "{req}": {meta_path}')

    meta: dict[str, Any] = {}
    meta["acquisitionID"] = _char_field(sm, "acquisition_ID", "")
    meta["objective"] = _char_field(sm, "objective", "")
    meta["umPerPix"] = float(sm["um_per_pix"])
    meta["zStepUm"] = float(sm["z_step_um"])
    meta["voxelSizeUm"] = [meta["zStepUm"], meta["umPerPix"], meta["umPerPix"]]
    meta["horizontalResolution"] = _numeric_field(sm, "horizontal_resolution", float("nan"))
    meta["verticalResolution"] = _numeric_field(sm, "vertical_resolution", float("nan"))
    if math.isnan(meta["horizontalResolution"]) or math.isnan(meta["verticalResolution"]):
        meta["sensorShape"] = [float("nan"), float("nan")]
    else:
        meta["sensorShape"] = [meta["verticalResolution"], meta["horizontalResolution"]]
    meta["zRange"] = _numeric_field(sm, "z_range", float("nan"))
    meta["scanning"] = _char_field(sm, "scanning", "")
    meta["destripe"] = _char_field(sm, "destripe", "")
    meta["destripeStatus"] = _char_field(sm, "destripe_status", "")
    meta["laserPower"] = _parse_laser_power(_find_field_by_prefix(sm, "laser"))
    meta["tiles"] = _parse_tiles(_get_field_safe(raw, "tiles", None))

    seq_path = root / "sequence.json"
    if seq_path.is_file():
        try:
            with open(seq_path, encoding="utf-8") as fh:
                seq_struct = json.load(fh)
        except ValueError as err:
            raise ValueError(f"Failed to parse {seq_path} as JSON: {err}") from err
        meta["sequence"] = _parse_sequence(seq_struct, seq_path)
    else:
        meta["sequence"] = None

    return meta
