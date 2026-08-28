"""Read values from a (legacy) Prairie View config file.

Port of +ndr/+format/+prairieview/readconfig.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.prairieview.configfilename import configfilename
from ndr.format.prairieview.readxml import readxml


def _sanitize(name: str) -> str:
    """Replace characters invalid in MATLAB field names with underscores.

    Space and parentheses become underscores, matching the original
    ``readprairieconfig`` behavior, so ``'Frame period (us)'`` becomes
    ``'Frame_period__us_'``. Kept as-is in Python so the two ports produce
    identically-named fields.
    """
    return name.replace(" ", "_").replace("(", "_").replace(")", "_")


def readconfig(filename: str | Path) -> dict[str, Any]:
    """Read a legacy Prairie Technologies ``.pcf`` config file.

    Returns a dict whose keys mirror the file's sections and parameters.
    Section and parameter names have spaces and parentheses replaced with
    underscores (so ``[Main]`` -> ``v["Main"]``, ``Frame period (us)`` ->
    ``v["Main"]["Frame_period__us_"]``). The special
    ``[Image TimeStamp (us)]`` section is read as a vector
    ``v["Image_TimeStamp__us_"]`` with one timestamp (microseconds) per image;
    this is the per-frame time source for ``ndr.reader.prairieview``.

    For Prairie View 2.2+ recordings the config is an ``.xml`` document; in
    that case this function delegates to
    :func:`ndr.format.prairieview.readxml` and sets ``v["is_xml"] = True``.

    This is a revised port of ``readprairieconfig.m`` from
    VH-Lab/vhlab-TwoPhoton-matlab (Platforms/PrairieView). Section termination
    is robust to CR, LF, and CRLF line endings.

    Parameters
    ----------
    filename : str or Path
        A directory, a config-file path, or any file in the recording
        directory; resolved with
        :func:`ndr.format.prairieview.configfilename`.
    """
    filename = configfilename(filename)

    if Path(filename).suffix.lower() == ".xml":
        v = readxml(filename)
        v["is_xml"] = True
        return v

    v: dict[str, Any] = {"is_xml": False}

    txt = Path(filename).read_text(errors="replace")
    lines = re.split(r"\r\n|\r|\n", txt)
    N = len(lines)

    i = 0
    while i < N:
        s = lines[i].strip()
        if not s or s[0] != "[":
            i += 1
            continue

        endb = s.find("]")
        if endb == -1:
            i += 1
            continue
        secname = s[1:endb].strip()

        if secname.lower() == "image timestamp (us)":
            # The following lines (one per image) hold '<label>=<timestamp_us>'.
            if "Main" not in v or "Total_images" not in v.get("Main", {}):
                raise ValueError(
                    f"Config {filename} has an [Image TimeStamp (us)] section but no "
                    "[Main] Total images count was read before it."
                )
            nimg = int(v["Main"]["Total_images"])
            ts = np.full(nimg, np.nan)
            for k in range(nimg):
                i += 1
                if i >= N:
                    break
                ln = lines[i]
                eqi = ln.find("=")
                if eqi != -1:
                    try:
                        ts[k] = float(ln[eqi + 1 :].strip())
                    except ValueError:
                        ts[k] = np.nan
            v["Image_TimeStamp__us_"] = ts
            i += 1
        else:
            subname = _sanitize(secname)
            field_struct: dict[str, Any] = {}
            i += 1
            while i < N:
                ln = lines[i].strip()
                if not ln or ln[0] == "[":
                    break  # blank line or next section ends this section
                if ln.count("=") > 1:
                    raise ValueError(
                        "Found more than one equal sign on a line in config " f"{filename}."
                    )
                eqi = ln.find("=")
                if eqi != -1:
                    field = _sanitize(ln[:eqi].strip())
                    rawval = ln[eqi + 1 :].strip()
                    try:
                        value: Any = float(rawval)
                    except ValueError:
                        value = rawval  # not a number; keep the string
                    field_struct[field] = value
                i += 1
            v[subname] = field_struct

    return v
