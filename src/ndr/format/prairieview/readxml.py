"""Read a Prairie View XML configuration file.

Port of +ndr/+format/+prairieview/readxml.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.prairieview.configfilename import configfilename
from ndr.format.prairieview.elementvalue import elementvalue
from ndr.format.prairieview.keyvalue import keyvalue


def readxml(filename: str | Path) -> dict[str, Any]:
    """Read a Prairie View XML parameter file.

    Returns a dict in the same shape as
    :func:`ndr.format.prairieview.readconfig` (the legacy ``.pcf`` reader), so
    the two formats are interchangeable for ``ndr.reader.prairieview``::

        v["Main"]["Lines_per_frame"]
        v["Main"]["Pixels_per_line"]
        v["Main"]["Frame_period__us_"]      # when available
        v["Main"]["Total_images"]
        v["Image_TimeStamp__us_"]           # per-frame timestamps, microseconds

    The per-frame timestamps are the real recorded times (not a uniform frame
    period): for modern PVScan files they are the ``<Frame absoluteTime>``
    values; for the older MM-era XML they are the per-frame ``<Time>`` values.

    This is a revised port of ``readprairieviewxml.m`` /
    ``readprairieviewxml3.m`` from VH-Lab/vhlab-TwoPhoton-matlab
    (Platforms/PrairieView). The tag names and timestamp semantics
    (``absoluteTime * 1e6`` for modern; ``Time * 1e3`` for legacy) are
    preserved.

    Parameters
    ----------
    filename : str or Path
        A directory, the XML file, or any file in the recording directory; the
        XML is resolved with :func:`ndr.format.prairieview.configfilename`.
    """
    filename = configfilename(filename)
    txt = Path(filename).read_text(errors="replace")

    if re.search(r'<PVScan[^>]*version="([^"]+)"', txt):
        return _read_modern(txt)
    return _read_legacy(txt)


def _read_modern(txt: str) -> dict[str, Any]:
    """Read a modern PVScan XML (vers 3/4/5; ported from readprairieviewxml3)."""
    v: dict[str, Any] = {"Main": {}}

    # Per-frame absolute times (seconds) -> microseconds, in file order.
    at = re.findall(r'<Frame[^>]*absoluteTime="([-+0-9.eE]+)"', txt)
    times = np.array([float(a) for a in at], dtype=float)

    v["Image_TimeStamp__us_"] = times * 1e6
    v["Main"]["Total_images"] = len(times)

    v["Main"]["Lines_per_frame"] = keyvalue(txt, "linesPerFrame")
    v["Main"]["Pixels_per_line"] = keyvalue(txt, "pixelsPerLine")

    fp = keyvalue(txt, "framePeriod")
    if isinstance(fp, float):
        v["Main"]["Frame_period__us_"] = fp * 1e6  # framePeriod is in seconds

    dt = keyvalue(txt, "dwellTime")
    if isinstance(dt, float):
        v["Main"]["Dwell_time__us_"] = dt

    # scanLinePeriod is the (exact) time to scan one line, in seconds.
    slp = keyvalue(txt, "scanLinePeriod")
    if isinstance(slp, float):
        v["Main"]["ScanLine_period__us_"] = slp * 1e6

    # bidirectionalScan is stored as a 'True'/'False' string.
    bd = keyvalue(txt, "bidirectionalScan")
    if isinstance(bd, str):
        v["Main"]["Bidirectional"] = bd.strip().lower() == "true"
    elif isinstance(bd, float):
        v["Main"]["Bidirectional"] = bool(bd)

    return v


def _read_legacy(txt: str) -> dict[str, Any]:
    """Read a legacy MM-era XML (ported from readprairieviewxml)."""
    v: dict[str, Any] = {"Main": {}}

    # Older Prairie XML (e.g. v2.2 '.NET DataSet' files) embeds an XSD schema
    # before the data; element names appear there as '<xs:element name="..."/>'
    # defining the fields. Strip the schema so values are read from the data
    # rows, not from the schema definitions.
    marker = "</xs:schema>"
    idx = txt.rfind(marker)
    if idx != -1:
        txt = txt[idx + len(marker) :]

    v["Main"]["Lines_per_frame"] = elementvalue(txt, "Lines_Per_Frame")
    v["Main"]["Pixels_per_line"] = elementvalue(txt, "Pixels_Per_Line")

    fr = elementvalue(txt, "Framerate")
    if isinstance(fr, float) and fr != 0:
        v["Main"]["Frame_period__us_"] = (1.0 / fr) * 1e6

    # One '<...Time...>VALUE<...>' (milliseconds) per '<Dataset_x0020_N>' frame
    # row, in file order.
    ts: list[float] = []
    for m in re.finditer(r"<Dataset_x0020_\d+>", txt):
        seg = txt[m.start() :]
        tm = re.search(r"<[^>]*Time[^>]*>([^<]*)<", seg)
        if tm:
            try:
                ts.append(float(tm.group(1)))
            except ValueError:
                ts.append(float("nan"))

    v["Image_TimeStamp__us_"] = np.array(ts, dtype=float) * 1e3  # ms -> us
    v["Main"]["Total_images"] = len(ts)

    return v
