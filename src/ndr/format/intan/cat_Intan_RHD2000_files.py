"""Concatenate multiple Intan RHD2000 files together.

Port of +ndr/+format/+intan/cat_Intan_RHD2000_files.m
"""

from __future__ import annotations

from pathlib import Path

from ndr.format.intan.read_Intan_RHD2000_datafile import _get_header_size
from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header

_COPY_CHUNK = 10000


def _headers_compatible(h1: dict, h2: dict, name1: str, name2: str) -> None:
    """Raise if two headers have incompatible acquisition parameters."""
    # Compare frequency_parameters
    fp1 = h1.get("frequency_parameters", {})
    fp2 = h2.get("frequency_parameters", {})
    if fp1 != fp2:
        raise ValueError(
            f"Acquisition parameters must be the same ({name1} != {name2}): "
            "frequency_parameters differ."
        )

    # Compare channel lists
    for key in (
        "amplifier_channels",
        "aux_input_channels",
        "supply_voltage_channels",
        "board_adc_channels",
        "board_dig_in_channels",
        "board_dig_out_channels",
    ):
        if h1.get(key, []) != h2.get(key, []):
            raise ValueError(
                f"Acquisition parameters must be the same ({name1} != {name2}): " f"{key} differ."
            )

    # Compare scalar header fields (excluding notes/version-specific info)
    for key in ("dc_amplifier_data_saved", "eval_board_mode"):
        if h1.get(key) != h2.get(key):
            raise ValueError(
                f"Acquisition parameters must be the same ({name1} != {name2}): " f"{key} differ."
            )


def cat_Intan_RHD2000_files(*filenames: str | Path) -> int:
    """Concatenate multiple RHD files together.

    Produces a large file named ``cat<FILENAME1>`` in the same directory as
    the first input file.  The header of the output file will be that of
    *FILENAME1*.

    Parameters
    ----------
    *filenames : str or Path
        Paths to the RHD files to concatenate.

    Returns
    -------
    int
        Status code (0 on success).

    Raises
    ------
    ValueError
        If acquisition parameters differ between files.
    """
    if len(filenames) == 0:
        return 0

    paths = [Path(f) for f in filenames]

    # Read all headers
    headers = [read_Intan_RHD2000_header(p) for p in paths]
    header_sizes = [_get_header_size(p, h) for p, h in zip(paths, headers)]

    # Verify all headers are compatible
    for i in range(len(headers)):
        for j in range(i + 1, len(headers)):
            _headers_compatible(headers[i], headers[j], str(paths[i]), str(paths[j]))

    # Build output filename: prepend "cat" to the first filename
    outfile = paths[0].parent / ("cat" + paths[0].name)

    with open(outfile, "wb") as fout:
        for i, path in enumerate(paths):
            with open(path, "rb") as fin:
                if i != 0:
                    # Skip header for all files after the first
                    fin.seek(header_sizes[i])
                while True:
                    chunk = fin.read(_COPY_CHUNK)
                    if not chunk:
                        break
                    fout.write(chunk)

    return 0
