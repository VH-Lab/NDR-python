"""Read header information from a Dabrowska lab MAT file.

Port of +ndr/+format/+dabrowska/header.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover
    sio = None  # type: ignore[assignment]


def header(filename: str | Path) -> dict[str, Any]:
    """Read the header (Pars variable) from a Dabrowska lab MAT file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.mat`` file.

    Returns
    -------
    dict
        The ``Pars`` structure from the MAT file, converted to a dict.
    """
    if sio is None:
        raise ImportError("scipy is required for reading MAT files.")

    filename = Path(filename)
    if filename.suffix != ".mat":
        raise ValueError(f'Expected file "{filename}" to have a ".mat" extension.')

    mat = sio.loadmat(str(filename), squeeze_me=True)
    if "Pars" not in mat:
        raise KeyError(f'File "{filename}" is missing the required variable "Pars".')

    pars = mat["Pars"]
    # Convert structured array to dict if needed
    if hasattr(pars, "dtype") and pars.dtype.names:
        result = {}
        for name in pars.dtype.names:
            val = pars[name]
            if hasattr(val, "item"):
                val = val.item()
            result[name] = val
        return result

    return dict(pars) if isinstance(pars, dict) else {"Pars": pars}
