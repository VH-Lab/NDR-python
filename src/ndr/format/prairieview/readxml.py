"""Read a Prairie View XML configuration file.

Port of +ndr/+format/+prairieview/readxml.m
"""

from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np

from ndr.format.prairieview.configfilename import configfilename
from ndr.format.prairieview.elementvalue import elementvalue
from ndr.format.prairieview.keyvalue import keyvalue


def readxml(filename: str) -> Dict[str, Any]:
    """Read a Prairie View XML parameter file.

    Reads a Prairie View XML parameter file and returns a dict ``v`` in the
    same shape as :func:`ndr.format.prairieview.readconfig.readconfig` (the
    legacy ``.pcf`` reader), so the two formats are interchangeable for
    :class:`ndr.reader.prairieview.ndr_reader_prairieview`::

        v["Main"]["Lines_per_frame"]
        v["Main"]["Pixels_per_line"]
        v["Main"]["Frame_period__us_"]      # when available
        v["Main"]["Total_images"]
        v["Image_TimeStamp__us_"]           # per-frame timestamps, microseconds

    The per-frame timestamps are the real recorded times (not a uniform frame
    period): for modern PVScan files they are the ``<Frame absoluteTime>``
    values; for the older MM-era XML they are the per-frame ``<Time>`` values.

    This is a faithful port of ``ndr.format.prairieview.readxml`` (itself a
    revised port of ``readprairieviewxml.m`` / ``readprairieviewxml3.m`` from
    VH-Lab/vhlab-TwoPhoton-matlab). The tag names and timestamp semantics
    (``absoluteTime`` * 1e6 for modern; ``Time`` * 1e3 for legacy) are
    preserved, with whole-file regular-expression parsing for robustness.

    Parameters
    ----------
    filename : str
        A directory, the XML file, or any file in the recording directory; the
        XML is resolved with :func:`configfilename`.

    Returns
    -------
    dict
        The parsed configuration (see shape above). The per-frame timestamps
        live under key ``"Image_TimeStamp__us_"`` as a ``numpy.ndarray``.
    """
    filename = configfilename(filename)
    with open(filename, "r") as f:
        txt = f.read()

    versiontok = re.search(r'<PVScan[^>]*version="([^"]+)"', txt)
    if versiontok is not None:
        return _local_read_modern(txt)
    return _local_read_legacy(txt)


# ----- modern PVScan (vers 3/4/5; ported from readprairieviewxml3) ---------


def _local_read_modern(txt: str) -> Dict[str, Any]:
    """Parse a modern PVScan (v3/4/5) Prairie View XML string."""
    v: Dict[str, Any] = {}
    # per-frame absolute times (seconds) -> microseconds, in file order
    at = re.findall(r'<Frame[^>]*absoluteTime="([-+0-9.eE]+)"', txt)
    times = np.array([float(x) for x in at], dtype=float)
    v["Image_TimeStamp__us_"] = times * 1e6
    v.setdefault("Main", {})
    v["Main"]["Total_images"] = int(times.size)

    v["Main"]["Lines_per_frame"] = keyvalue(txt, "linesPerFrame")
    v["Main"]["Pixels_per_line"] = keyvalue(txt, "pixelsPerLine")
    fp = keyvalue(txt, "framePeriod")
    if fp is not None and isinstance(fp, (int, float)) and not isinstance(fp, bool):
        v["Main"]["Frame_period__us_"] = fp * 1e6  # framePeriod is in seconds
    dt = keyvalue(txt, "dwellTime")
    if dt is not None and isinstance(dt, (int, float)) and not isinstance(dt, bool):
        v["Main"]["Dwell_time__us_"] = dt
    return v


# ----- legacy MM-era XML (ported from readprairieviewxml) -------------------


def _local_read_legacy(txt: str) -> Dict[str, Any]:
    """Parse a legacy MM-era ('.NET DataSet') Prairie View XML string."""
    v: Dict[str, Any] = {}

    # Older Prairie XML (e.g. v2.2 '.NET DataSet' files) embeds an XSD schema
    # before the data; element names appear there as '<xs:element name="..."/>'
    # defining the fields. Strip the schema so values are read from the data
    # rows, not from the schema definitions.
    schema_marker = "</xs:schema>"
    idx = txt.rfind(schema_marker)
    if idx != -1:
        txt = txt[idx + len(schema_marker):]

    v.setdefault("Main", {})
    v["Main"]["Lines_per_frame"] = elementvalue(txt, "Lines_Per_Frame")
    v["Main"]["Pixels_per_line"] = elementvalue(txt, "Pixels_Per_Line")
    fr = elementvalue(txt, "Framerate")
    if (
        fr is not None
        and isinstance(fr, (int, float))
        and not isinstance(fr, bool)
        and fr != 0
    ):
        v["Main"]["Frame_period__us_"] = (1.0 / fr) * 1e6

    # one '<...Time...>VALUE<...>' (milliseconds) per '<Dataset_x0020_N>' frame
    # row, in file order
    starts = [m.start() for m in re.finditer(r"<Dataset_x0020_\d+>", txt)]
    ts_list = []
    for s in starts:
        seg = txt[s:]
        tm = re.search(r"<[^>]*Time[^>]*>([^<]*)<", seg)
        if tm is not None:
            ts_list.append(float(tm.group(1)))
    ts = np.array(ts_list, dtype=float)
    v["Image_TimeStamp__us_"] = ts * 1e3  # milliseconds -> microseconds
    v["Main"]["Total_images"] = int(ts.size)
    return v
