"""Detect whether an RHD file is part of a multi-file recording.

Port of ``+ndr/+format/+intan/detectRHD2000FileMode.m``.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Intan's acquisition software names the members of a split recording
#: ``<prefix>_<YYMMDD>_<HHMMSS>.rhd``. Both groups are exactly six digits;
#: a name that does not match this shape is not part of a set.
_TIMESTAMPED = re.compile(r"^(.*)_(\d{6})_(\d{6})$")


def detectRHD2000FileMode(filename: str | Path) -> str:
    """Return ``"multiFile"`` or ``"singleFile"`` for ``filename``.

    ``"multiFile"`` when FILENAME follows the Intan
    ``<prefix>_<YYMMDD>_<HHMMSS>.rhd`` naming convention AND at least one
    additional sibling file in the same directory shares that prefix and
    matches the same timestamp pattern. ``"singleFile"`` otherwise -- so a
    lone timestamped file is single, since one file is not a set.

    Parameters
    ----------
    filename : str or Path
        Path to an .rhd file. It need not exist; only its name and its
        directory's contents are consulted.

    Returns
    -------
    str
        ``"multiFile"`` or ``"singleFile"``.

    See Also
    --------
    getRHD2000FileList, read_Intan_RHD2000_header, read_Intan_RHD2000_datafile
    """
    path = Path(filename)
    dirname = path.parent if str(path.parent) else Path.cwd()
    match = _TIMESTAMPED.match(path.stem)
    if match is None:
        return "singleFile"

    prefix = match.group(1)
    sibling = re.compile(rf"^{re.escape(prefix)}_(\d{{6}})_(\d{{6}})$")

    count = 0
    try:
        candidates = sorted(dirname.glob(f"{glob_escape(prefix)}_*_*{path.suffix}"))
    except OSError:  # pragma: no cover - unreadable directory
        return "singleFile"
    for candidate in candidates:
        if sibling.match(candidate.stem):
            count += 1
            if count > 1:
                return "multiFile"
    return "singleFile"


def glob_escape(text: str) -> str:
    """Escape glob metacharacters in a literal prefix.

    MATLAB's ``dir`` takes the prefix literally apart from ``*``; Python's
    ``Path.glob`` also honours ``?`` and ``[]``, so a recording whose prefix
    contains one of those would otherwise match the wrong siblings. The
    regex re-check below would reject them anyway, but escaping keeps the
    glob and the regex describing the same set.
    """
    return "".join(f"[{ch}]" if ch in "*?[]" else ch for ch in text)
