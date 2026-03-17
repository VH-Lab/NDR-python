"""Read log files from a TDT SEV directory.

Port of +ndr/+format/+tdt/read_SEV_logs.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def read_SEV_logs(dirname: str | Path, *, VERBOSE: bool = False) -> list[dict[str, Any]]:
    """Read the ``*_log.txt`` files in a TDT SEV directory.

    Parameters
    ----------
    dirname : str or Path
        Path to the directory containing the SEV recording.
    VERBOSE : bool
        Print log contents to stdout.

    Returns
    -------
    list of dict
        Sample info entries with fields ``name``, ``start_sample``,
        ``hour``, ``gaps``, ``gap_text``.
    """
    dirname = Path(dirname)
    sample_info: list[dict[str, Any]] = []

    txt_files = sorted(dirname.glob("*_log.txt"))
    if not txt_files and VERBOSE:
        print(f"info: no log files in {dirname}")
        return sample_info

    for txt_file in txt_files:
        if VERBOSE:
            print(f"info: log file {txt_file.name}")

        matches = re.match(r"^[^_|-]+(?=_|-)", txt_file.stem)
        if not matches:
            continue

        entry: dict[str, Any] = {"name": matches.group(0)}
        log_text = txt_file.read_text()
        if VERBOSE:
            print(log_text)

        t = re.search(r"recording started at sample: (\d+)", log_text)
        entry["start_sample"] = int(t.group(1)) if t else 1

        hour_match = re.search(r"-(\d)h", txt_file.name)
        entry["hour"] = int(hour_match.group(1)) if hour_match else 0

        if entry["start_sample"] > 2 and entry["hour"] == 0:
            raise RuntimeError(
                f"{entry['name']} store starts on sample {entry['start_sample']}"
            )

        # Look for gap info
        entry["gaps"] = []
        entry["gap_text"] = ""
        gap_matches = re.findall(
            r"gap detected. last saved sample: (\d+), new saved sample: (\d+)",
            log_text,
        )
        if gap_matches:
            entry["gaps"] = [(int(a), int(b)) for a, b in gap_matches]
            entry["gap_text"] = "; ".join(
                [f"last={a}, new={b}" for a, b in gap_matches]
            )
            if entry["hour"] > 0:
                raise RuntimeError(
                    f"gaps detected in data set for {entry['name']}-{entry['hour']}h!\n"
                    f"   {entry['gap_text']}\nContact TDT for assistance."
                )
            else:
                raise RuntimeError(
                    f"gaps detected in data set for {entry['name']}!\n"
                    f"   {entry['gap_text']}\nContact TDT for assistance."
                )

        sample_info.append(entry)

    return sample_info
