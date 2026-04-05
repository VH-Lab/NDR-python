"""Read and parse a SpikeGLX .meta file into a dictionary.

Port of +ndr/+format/+neuropixelsGLX/readmeta.m
"""

from __future__ import annotations

from pathlib import Path


def readmeta(metafilename: str | Path) -> dict[str, str]:
    """Read and parse a SpikeGLX .meta file into a dictionary.

    Reads a SpikeGLX metadata file (.ap.meta or .lf.meta) and returns a
    dictionary where each key=value line in the file becomes an entry.
    All values are kept as strings; numeric conversion is left to the caller.

    Parameters
    ----------
    metafilename : str or Path
        Full path to the .meta file.

    Returns
    -------
    dict of str to str
        Keys and values from the .meta file. Tildes and dots in key names
        are replaced with underscores.
    """
    metafilename = Path(metafilename)
    if not metafilename.is_file():
        raise FileNotFoundError(f"Could not open meta file: {metafilename}")

    meta: dict[str, str] = {}

    with open(metafilename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eq_pos = line.find("=")
            if eq_pos < 0:
                continue
            key = line[:eq_pos].strip()
            value = line[eq_pos + 1 :].strip()

            # Replace characters that are invalid in Python identifiers
            key = key.replace("~", "_").replace(".", "_")

            if key:
                meta[key] = value

    return meta
