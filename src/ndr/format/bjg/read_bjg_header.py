"""Read header information from a BJG .bin file.

Port of +ndr/+format/+bjg/read_bjg_header.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_bjg_header(filename: str | Path) -> dict[str, Any]:
    """Read header information from a BJG binary file.

    Parameters
    ----------
    filename : str or Path
        Path to the BJG ``.bin`` file.

    Returns
    -------
    dict
        Header information with fields: ``header_size``, ``data_size``,
        ``bytes_per_sample``, ``format``, ``datestamp``, ``num_channels``,
        ``sample_rate``, ``channel_names``, ``samples``, ``local_t0``,
        ``local_t1``, ``duration``.
    """
    filename = Path(filename)
    header_length = 4096

    with open(filename, "rb") as f:
        hdr_bytes = f.read(header_length)

    hdr_text = hdr_bytes.decode("ascii", errors="replace")
    lines = hdr_text.split("\n")

    h: dict[str, Any] = {}
    h["header_size"] = 4096
    h["data_size"] = filename.stat().st_size - h["header_size"]
    h["bytes_per_sample"] = 4

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if "BJG" in line:
            h["format"] = line

        # Check for datestamp pattern (has 2 colons and 3 hyphens)
        if line.count(":") == 2 and line.count("-") == 3:
            h["datestamp"] = line

        if "Channels" in line:
            import re
            m = re.match(r"(\d+)\s*Channels", line)
            if m:
                h["num_channels"] = int(m.group(1))

        if "Samples/Second per Channel" in line:
            import re
            m = re.match(r"([\d.]+)\s*Samples/Second per Channel", line)
            if m:
                h["sample_rate"] = float(m.group(1))

        if line.lower() == "start":
            channel_names = []
            i += 1
            while i < len(lines):
                cline = lines[i].strip()
                if cline.lower() == "stop":
                    break
                if cline:
                    channel_names.append(cline)
                i += 1
            h["channel_names"] = channel_names

        i += 1

    h["samples"] = h["data_size"] // (h["num_channels"] * h["bytes_per_sample"])
    h["local_t0"] = 0.0
    h["local_t1"] = (1.0 / h["sample_rate"]) * (h["samples"] - 1)
    h["duration"] = (1.0 / h["sample_rate"]) * h["samples"]

    return h
