"""Read an Intan RHD2000 data file and print a summary.

Port of +ndr/+format/+intan/+manufacturer/read_Intan_RHD2000_file.m

The original MATLAB version is a script that reads data and moves variables
into the MATLAB workspace.  This Python port wraps
:func:`read_Intan_RHD2000_file_var` and prints a human-readable summary,
returning the same result dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ndr.format.intan.manufacturer.read_Intan_RHD2000_file_var import (
    read_Intan_RHD2000_file_var,
)


def _plural(n: int) -> str:
    """Return 's' when *n* != 1."""
    return "" if n == 1 else "s"


def read_Intan_RHD2000_file(filename: str | Path) -> dict[str, Any]:
    """Read an Intan RHD2000 data file and print a summary.

    This is equivalent to the manufacturer's ``read_Intan_RHD2000_file.m``
    script.  It reads all data from the file and prints a summary of the
    contents to stdout.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.rhd`` data file.

    Returns
    -------
    dict
        Dictionary containing header information, channel metadata, and all
        recorded data arrays (same as :func:`read_Intan_RHD2000_file_var`).
    """
    result = read_Intan_RHD2000_file_var(filename)

    fp = result["frequency_parameters"]
    print(f"Sample rate: {fp['amplifier_sample_rate']:.2f} Hz")

    n_amp = len(result.get("amplifier_channels", []))
    n_aux = len(result.get("aux_input_channels", []))
    n_sv = len(result.get("supply_voltage_channels", []))
    n_adc = len(result.get("board_adc_channels", []))
    n_din = len(result.get("board_dig_in_channels", []))
    n_dout = len(result.get("board_dig_out_channels", []))

    print(f"{n_amp} amplifier channel{_plural(n_amp)}")
    print(f"{n_aux} auxiliary input channel{_plural(n_aux)}")
    print(f"{n_sv} supply voltage channel{_plural(n_sv)}")
    print(f"{n_adc} board ADC channel{_plural(n_adc)}")
    print(f"{n_din} board digital input channel{_plural(n_din)}")
    print(f"{n_dout} board digital output channel{_plural(n_dout)}")

    if "t_amplifier" in result:
        t = result["t_amplifier"]
        duration = t[-1] - t[0] if len(t) > 1 else 0.0
        print(f"Duration: {duration:.2f} seconds")

    return result
