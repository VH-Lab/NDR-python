"""Read values from a (legacy) Prairie View config file.

Port of +ndr/+format/+prairieview/readconfig.m
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from ndr.format.prairieview.configfilename import configfilename
from ndr.format.prairieview.keyvalue import _str2double
from ndr.format.prairieview.readxml import readxml


def readconfig(filename: str) -> dict[str, Any]:
    """Read values from a (legacy) Prairie View config file.

    Reads a legacy Prairie Technologies ``.pcf`` config file and returns a
    dict ``v`` whose keys mirror the file's sections and parameters. Section
    and parameter names have spaces and parentheses replaced with underscores
    (so ``[Main]`` -> ``v["Main"]``, ``Frame period (us)`` ->
    ``v["Main"]["Frame_period__us_"]``). The special ``[Image TimeStamp (us)]``
    section is read as a vector ``v["Image_TimeStamp__us_"]`` with one timestamp
    (microseconds) per image; this is the per-frame time source for
    :class:`ndr.reader.prairieview.ndr_reader_prairieview`.

    For Prairie View 2.2+ recordings the config is an ``.xml`` document; in that
    case this function delegates to :func:`ndr.format.prairieview.readxml` and
    sets ``v["is_xml"] = True``.

    This is a faithful port of ``ndr.format.prairieview.readconfig`` (itself a
    revised port of ``readprairieconfig.m`` from VH-Lab/vhlab-TwoPhoton-matlab).
    The ``.pcf`` parsing behavior is preserved, with section termination robust
    to CR, LF, and CRLF line endings.

    Parameters
    ----------
    filename : str
        A directory, a config-file path, or any file in the recording
        directory; the config file is resolved with :func:`configfilename`.

    Returns
    -------
    dict
        The parsed configuration. Always contains key ``"is_xml"``.

    Raises
    ------
    ValueError
        If an ``[Image TimeStamp (us)]`` section is present but no
        ``[Main] Total images`` count was read before it, or if a parameter
        line has more than one ``=`` sign.
    """
    filename = configfilename(filename)

    v: dict[str, Any] = {}
    ext = os.path.splitext(filename)[1]
    if ext.lower() == ".xml":
        v = readxml(filename)
        v["is_xml"] = True
        return v
    v["is_xml"] = False

    with open(filename) as f:
        txt = f.read()
    lines = re.split(r"\r\n|\r|\n", txt)
    N = len(lines)

    i = 0
    while i < N:
        s = lines[i].strip()
        if s == "" or s[0] != "[":
            i += 1
            continue

        endb = s.find("]")
        if endb == -1:
            i += 1
            continue
        secname = s[1:endb].strip()

        if secname.lower() == "image timestamp (us)":
            # the following lines (one per image) hold '<label>=<timestamp_us>'
            if "Main" not in v or "Total_images" not in v["Main"]:
                raise ValueError(
                    f"Config {filename} has an [Image TimeStamp (us)] section "
                    "but no [Main] Total images count was read before it."
                )
            nimg = int(v["Main"]["Total_images"])
            ts = np.full(nimg, np.nan, dtype=float)
            for k in range(nimg):
                i += 1
                if i >= N:
                    break
                ln = lines[i]
                eqi = ln.find("=")
                if eqi != -1:
                    parsed = _str2double(ln[eqi + 1 :].strip())
                    ts[k] = np.nan if parsed is None else parsed
            v["Image_TimeStamp__us_"] = ts
            i += 1
        else:
            subname = _local_sanitize(secname)
            field_struct: dict[str, Any] = {}
            i += 1
            while i < N:
                ln = lines[i].strip()
                if ln == "" or ln[0] == "[":
                    break  # blank line or next section ends this section
                positions = [m.start() for m in re.finditer("=", ln)]
                if len(positions) > 1:
                    raise ValueError(
                        "Found more than one equal sign on a line in config " f"{filename}."
                    )
                if positions:
                    eqi = positions[0]
                    field = _local_sanitize(ln[:eqi].strip())
                    rawval = ln[eqi + 1 :].strip()
                    val = _str2double(rawval)
                    if val is None:  # not a number; keep the string
                        val = rawval
                    field_struct[field] = val
                i += 1
            v[subname] = field_struct

    return v


def _local_sanitize(name: str) -> str:
    """Replace characters invalid in field names (space, parentheses) with '_'.

    Matches the original ``readprairieconfig`` ``localsanitize`` behavior.
    """
    return name.replace(" ", "_").replace("(", "_").replace(")", "_")
