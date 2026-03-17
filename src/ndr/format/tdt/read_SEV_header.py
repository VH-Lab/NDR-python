"""Read header and channel information from a TDT SEV directory.

Port of +ndr/+format/+tdt/read_SEV_header.m
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.tdt.read_SEV_logs import read_SEV_logs

_ALLOWED_FORMATS = ["single", "int32", "int16", "int8", "double", "int64"]
_FORMAT_TO_DTYPE = {
    "single": np.float32,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "double": np.float64,
    "int64": np.int64,
}


def read_SEV_header(
    dirname: str | Path, *, VERBOSE: bool = False, FS: float = 0
) -> list[dict[str, Any]]:
    """Read header info for a TDT directory of SEV files.

    Parameters
    ----------
    dirname : str or Path
        Path to the SEV directory.
    VERBOSE : bool
        Print verbose output.
    FS : float
        Override sampling rate (0 means use file header).

    Returns
    -------
    list of dict
        One entry per SEV file with fields including ``name``, ``chan``,
        ``hour``, ``fs``, ``dForm``, ``npts``, ``duration_seconds``, etc.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        raise ValueError(f"Expected a directory for {dirname}.")

    sample_info = read_SEV_logs(dirname, VERBOSE=VERBOSE)
    sev_files = sorted(dirname.glob("*.sev"))

    file_list = []
    for sev_path in sev_files:
        entry: dict[str, Any] = {
            "name": sev_path.name,
            "bytes": sev_path.stat().st_size,
        }

        name_no_ext = sev_path.stem

        # Find channel number
        ch_matches = re.findall(r"_[Cc]h(\d+)", name_no_ext)
        entry["chan"] = int(ch_matches[-1]) if ch_matches else 0

        # Find starting hour
        hour_matches = re.findall(r"-(\d+)h", name_no_ext)
        entry["hour"] = int(hour_matches[-1]) if hour_matches else 0

        entry["data_size"] = entry["bytes"] - 40

        with open(sev_path, "rb") as fid:
            stream_header: dict[str, Any] = {}
            stream_header["fileSizeBytes"] = struct.unpack("<Q", fid.read(8))[0]
            stream_header["fileType"] = fid.read(3).decode("ascii", errors="replace")
            stream_header["fileVersion"] = struct.unpack("B", fid.read(1))[0]

            # Event name from filename
            parts = name_no_ext.split("_")
            parts = [p for p in parts if p]
            if len(parts) > 1:
                nm = (parts[-2] + "____")[:4]
                stream_header["eventName"] = nm
            else:
                stream_header["eventName"] = name_no_ext

            if stream_header["fileVersion"] < 4:
                if stream_header["fileVersion"] == 3:
                    stream_header["eventName"] = fid.read(4).decode("ascii", errors="replace")
                else:
                    _old_name = fid.read(4).decode("ascii", errors="replace")

                entry["chan"] = struct.unpack("<H", fid.read(2))[0]
                stream_header["totalNumChannels"] = struct.unpack("<H", fid.read(2))[0]
                stream_header["sampleWidthBytes"] = struct.unpack("<H", fid.read(2))[0]
                _reserved = struct.unpack("<H", fid.read(2))[0]

                dform_byte = struct.unpack("B", fid.read(1))[0]
                stream_header["dForm"] = _ALLOWED_FORMATS[dform_byte & 7]

                decimate = struct.unpack("B", fid.read(1))[0]
                rate = struct.unpack("<H", fid.read(2))[0]

                stream_header["fs"] = 2 ** (rate - 12) * 25000000 / decimate
            else:
                raise ValueError(
                    f"{sev_path.name} has unknown version {stream_header['fileVersion']}"
                )

            if stream_header["fileVersion"] == 0:
                stream_header["dForm"] = "single"
                stream_header["fs"] = 24414.0625

        if FS > 0:
            stream_header["fs"] = FS

        # Add log info
        entry["start_sample"] = 1
        entry["gaps"] = []
        entry["gap_text"] = ""
        for si in sample_info:
            if si["name"] == stream_header["eventName"] and entry["hour"] == si.get("hour", 0):
                entry["start_sample"] = si["start_sample"]
                entry["gaps"] = si["gaps"]
                entry["gap_text"] = si["gap_text"]

        dt = _FORMAT_TO_DTYPE.get(stream_header["dForm"], np.float32)
        entry["itemSize"] = np.dtype(dt).itemsize
        entry["npts"] = entry["data_size"] // entry["itemSize"]
        entry["fs"] = stream_header["fs"]
        entry["dForm"] = stream_header["dForm"]
        entry["eventName"] = stream_header["eventName"]
        entry["duration_seconds"] = entry["npts"] / entry["fs"] if entry["fs"] > 0 else 0

        # Parse start date/time from filename
        sp = sev_path.name.split("-")
        if len(sp) == 5:
            entry["start_date"] = "20" + sp[3]
            entry["start_time"] = sp[4][:6]
        else:
            entry["start_date"] = ""
            entry["start_time"] = ""

        file_list.append(entry)

    return file_list
