"""Locate the (legacy) Prairie View configuration file.

Port of +ndr/+format/+prairieview/configfilename.m
"""

from __future__ import annotations

from pathlib import Path

# Helper files that are not the recording's config. The original excluded
# 'tour.xml' and the misspelled 'exlude.xml'; 'exclude.xml' is allowed for too.
_SKIP_XML = {"tour.xml", "exlude.xml", "exclude.xml"}


def configfilename(dirname: str | Path) -> str:
    """Return the name of the (legacy) Prairie View configuration file.

    If ``dirname`` is a directory, locates the recording's config file within
    it: a legacy ``*_Main.pcf`` if present, otherwise a ``*.xml`` (excluding
    the ``tour.xml`` / ``exclude.xml`` helper files). If a file is passed, its
    parent directory is searched the same way.

    This is a revised port of ``tpconfigfilename.m`` from
    VH-Lab/vhlab-TwoPhoton-matlab (Platforms/PrairieView).

    Parameters
    ----------
    dirname : str or Path
        A recording directory, a config file, or any file in the recording
        directory.

    Returns
    -------
    str
        Path to the resolved config file.
    """
    p = Path(dirname)
    tpdir = p if p.is_dir() else (p.parent if str(p.parent) else Path.cwd())
    if not str(tpdir):
        tpdir = Path.cwd()

    pcfiles = sorted(f for f in tpdir.glob("*_Main.pcf") if f.is_file())
    if pcfiles:
        return str(pcfiles[-1])

    xmls = sorted(f for f in tpdir.glob("*.xml") if f.is_file())
    include = [f for f in xmls if f.name.lower() not in _SKIP_XML]

    if not include:
        raise FileNotFoundError(
            f"Could not find a Prairie config file (*_Main.pcf or *.xml) for {dirname}."
        )

    return str(include[-1])
