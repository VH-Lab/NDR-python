"""List the RHD2000 files that make up one recording.

Port of ``+ndr/+format/+intan/getRHD2000FileList.m``.
"""

from __future__ import annotations

import re
from pathlib import Path

from ndr.format.intan.detectRHD2000FileMode import _TIMESTAMPED, detectRHD2000FileMode, glob_escape


def getRHD2000FileList(filename: str | Path, fileMode: str = "detect") -> list[str]:
    """Return the full-path files that together make up a recording.

    Parameters
    ----------
    filename : str or Path
        A single .rhd file belonging to the recording.
    fileMode : str, optional
        ``"detect"`` (default) resolves to ``"multiFile"`` or
        ``"singleFile"`` via :func:`detectRHD2000FileMode`.
        ``"singleFile"`` returns just FILENAME even when siblings exist.
        ``"multiFile"`` returns the chronologically sorted list of every
        file in the same directory matching
        ``<prefix>_<YYMMDD>_<HHMMSS>.rhd``, where ``<prefix>`` is parsed
        from FILENAME.

    Returns
    -------
    list of str
        Full paths, chronologically ordered in multi-file mode.

    Raises
    ------
    ValueError
        If FILENAME does not match the Intan multi-file pattern while
        ``fileMode`` is ``"multiFile"``, if no matching set is found, or if
        ``fileMode`` is not one of the three accepted values.

    See Also
    --------
    detectRHD2000FileMode, read_Intan_RHD2000_header, read_Intan_RHD2000_datafile
    """
    path = Path(filename)

    if fileMode == "detect":
        fileMode = detectRHD2000FileMode(path)

    if fileMode == "singleFile":
        return [str(path)]

    if fileMode != "multiFile":
        raise ValueError(
            f"Unknown fileMode: {fileMode}. Use 'detect', 'singleFile' or 'multiFile'."
        )

    dirname = path.parent if str(path.parent) else Path.cwd()
    match = _TIMESTAMPED.match(path.stem)
    if match is None:
        raise ValueError(
            f"Filename {path} does not match the Intan multi-file pattern "
            f"<prefix>_YYMMDD_HHMMSS{path.suffix}."
        )
    prefix = match.group(1)
    sibling = re.compile(rf"^{re.escape(prefix)}_(\d{{6}})_(\d{{6}})$")

    # Sort on the concatenated YYMMDD+HHMMSS exactly as MATLAB does. That is
    # a lexicographic sort on a two-digit year, so it orders correctly within
    # a century but not across 99 -> 00. Mirrored deliberately: the Lead-Follow
    # rule makes MATLAB the source of truth, and silently "fixing" the ordering
    # here would put the two languages out of step on the same directory.
    found: list[tuple[str, Path]] = []
    for candidate in dirname.glob(f"{glob_escape(prefix)}_*_*{path.suffix}"):
        hit = sibling.match(candidate.stem)
        if hit is not None:
            found.append((hit.group(1) + hit.group(2), candidate))

    if not found:
        raise ValueError(f"No RHD multi-file set found for prefix {prefix} in {dirname}.")

    found.sort(key=lambda item: item[0])
    return [str(candidate) for _, candidate in found]
