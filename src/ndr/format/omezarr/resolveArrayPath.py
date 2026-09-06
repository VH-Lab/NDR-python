"""Map (pyramid, level) to a Zarr array directory.

Port of ``+ndr/+format/+omezarr/resolveArrayPath.m``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import validate_call

from ndr.format.omezarr.listPyramids import listPyramids

__all__ = ["resolveArrayPath"]


def _format_name_list(names: list[str]) -> str:
    if not names:
        return "(none)"
    parts = [f'"{n}"' if n else '""' for n in names]
    return ", ".join(parts)


@validate_call
def resolveArrayPath(zarrPath: str, pyramidName: str, level: int) -> str:
    """Return the absolute on-disk directory of one pyramid level's Zarr array.

    ``pyramidName`` is one of the names returned by
    :func:`ndr.format.omezarr.listPyramids`. If the name is not present,
    this errors with a message listing what is available. This is the
    silent-failure guard: a reader cannot get the wrong pyramid because two
    entries were named the same, or because the caller misspelled one; it
    gets an error naming the choice.

    ``level`` is 1-based (level 1 is the highest resolution -- the first
    entry in the pyramid's ``datasets`` array). This matches MATLAB
    indexing and NDR convention; the NGFF file itself often labels the
    full-resolution dataset "0", but that is a path string, not an index.

    The shared-level-0 case (both pyramids' first entry points at the same
    on-disk array) resolves correctly: calling this for ``('mean', 1)`` and
    ``('max', 1)`` on the lab layout returns the same directory, because
    both pyramids' ``datasets[0].path`` is '0'.
    """
    pyramids = listPyramids(zarrPath)
    names = [p["name"] for p in pyramids]

    matches = [i for i, n in enumerate(names) if n == pyramidName]
    if not matches:
        raise ValueError(
            f'Pyramid "{pyramidName}" is not in {zarrPath}. '
            f"Available pyramids: {_format_name_list(names)}."
        )
    idx = matches[0]

    if level < 1:
        raise ValueError(f"level must be a positive integer; got {level}.")

    levels = pyramids[idx]["levels"]
    if level > len(levels):
        raise IndexError(
            f'Pyramid "{pyramidName}" has {len(levels)} level(s); ' f"level {level} requested."
        )

    path_str = levels[level - 1]["path"]
    array_path = Path(zarrPath) / path_str.replace("/", "/")

    if not array_path.is_dir():
        raise FileNotFoundError(
            f'Metadata for pyramid "{pyramidName}" level {level} points at '
            f'"{array_path}", but that directory does not exist.'
        )
    return str(array_path)
