"""Name of the (legacy) Prairie View configuration file.

Port of +ndr/+format/+prairieview/configfilename.m
"""

from __future__ import annotations

import glob
import os


def configfilename(dirname: str) -> str:
    """Return the path to the (legacy) Prairie View configuration file.

    If ``dirname`` is a directory, locate the recording's config file within
    it: a legacy ``*_Main.pcf`` if present, otherwise a ``*.xml`` (excluding
    the ``tour.xml`` / ``exclude.xml`` helper files). If a file is passed, its
    parent directory is searched the same way.

    This is a faithful port of ``ndr.format.prairieview.configfilename`` (itself
    a revised port of ``tpconfigfilename.m`` from
    VH-Lab/vhlab-TwoPhoton-matlab).

    Parameters
    ----------
    dirname : str
        A directory, a config-file path, or any file in the recording
        directory.

    Returns
    -------
    str
        The full path to the resolved Prairie View config file.

    Raises
    ------
    FileNotFoundError
        If no ``*_Main.pcf`` or eligible ``*.xml`` config file can be found.
    """
    if os.path.isdir(dirname):
        tpdir = dirname
    else:
        tpdir = os.path.dirname(dirname)
        if tpdir == "":
            tpdir = os.getcwd()

    pcfile = sorted(glob.glob(os.path.join(tpdir, "*_Main.pcf")))
    if not pcfile:
        xmls = sorted(glob.glob(os.path.join(tpdir, "*.xml")))
        include = []
        for x in xmls:
            nm = os.path.basename(x).lower()
            if nm in ("tour.xml", "exlude.xml", "exclude.xml"):
                # skip helper files (the original excluded 'tour.xml' and the
                # misspelled 'exlude.xml'; 'exclude.xml' is allowed for too)
                continue
            include.append(x)
        if not include:
            raise FileNotFoundError(
                "Could not find a Prairie config file (*_Main.pcf or *.xml) " f"for {dirname}."
            )
        return include[-1]
    return pcfile[-1]
