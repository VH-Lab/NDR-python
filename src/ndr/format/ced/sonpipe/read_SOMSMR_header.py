"""Read a CED Spike2 header through the sonpipe CLI.

Port of +ndr/+format/+ced/+sonpipe/read_SOMSMR_header.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ndr.format.ced.sonpipe._invoke import invoke_json


def read_SOMSMR_header(filename: str | Path) -> dict[str, Any]:
    """Return the file header as ``{"fileinfo": ..., "channelinfo": [...]}``.

    The compatibility aliases mirror read_SOMSMR_header.m: callers written
    against the classic SON header expect usPerTime, dTimeBase and maxFTime,
    which the sonpipe header spells differently.
    """
    raw = invoke_json(["header", str(filename)])

    fileinfo = dict(raw.get("fileinfo") or {})
    fileinfo["usPerTime"] = 1
    fileinfo["dTimeBase"] = fileinfo.get("timebase")
    fileinfo["maxFTime"] = fileinfo.get("max_time_ticks")

    return {"fileinfo": fileinfo, "channelinfo": list(raw.get("channelinfo") or [])}
