"""Read VHLV (VH Lab LabView) header files.

Port of +ndr/+format/+vld/readvhlvheaderfile.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _coerce(value: str) -> Any:
    """Coerce a header field value to a number when possible.

    MATLAB evaluates the field text and falls back to the raw string when
    that fails; here an int/float parse plays the same role for the field
    values this format actually carries.
    """
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def readvhlvheaderfile(myfilename: str | Path) -> dict[str, Any]:
    """Read the VHLV (VH Lab LabView) header file format.

    Reads the header file format for the VHLAB LabView multichannel
    acquisition system.

    ``myfilename`` is a text file (extension ``.vlh``) where each line begins
    with a field name followed by a colon and a tab, followed by the value.
    The expected fields are:

    ``ChannelString``
        The channel names acquired in the LabView system. The channel numbers
        correspond to these inputs; e.g. ``'/dev/ai0'`` means one channel,
        analog input 0 on the acquisition device.
    ``NumChans``
        The number of channels acquired.
    ``SamplingRate``
        The sampling rate of each channel, in Hz.
    ``SamplesPerChunk``
        How many samples were written to disk in each burst of recording.
    ``Multiplexed``
        Whether adjacent samples are from different channels (1) or the
        channel data is stored in groups of ``SamplesPerChunk`` (0).

    Use :func:`ndr.format.vld.readvhlvdatafile` to read the data.

    Parameters
    ----------
    myfilename : str or Path
        Path to the ``.vlh`` header file.

    Returns
    -------
    dict
        The header fields. Values parse to numbers where possible, otherwise
        they stay strings.
    """
    myfilename = Path(myfilename)

    if myfilename.suffix.lower() == ".vld":
        raise ValueError(
            f"It appears you are trying to open a data file {myfilename} "
            "with the code that reads the header."
        )

    try:
        text = myfilename.read_text()
    except OSError as err:
        raise OSError(f"Could not open file {myfilename}.") from err

    # Add a line feed to the beginning, ensure the last line ends in one, and
    # drop carriage returns (redundant with line feeds).
    text = "\n" + text
    if not text.endswith("\n"):
        text += "\n"
    text = text.replace("\r", "")

    # A colon followed by a tab marks a field: everything left of the colon on
    # that line is the field name, and the value runs from after the tab until
    # the line before the next such separator.
    sep = ":\t"
    separators = []
    pos = text.find(sep)
    while pos != -1:
        separators.append(pos)
        pos = text.find(sep, pos + 1)

    linefeeds = [i for i, ch in enumerate(text) if ch == "\n"]

    mystruct: dict[str, Any] = {}

    for i in range(1, len(linefeeds)):
        z = [s for s in separators if linefeeds[i - 1] < s < linefeeds[i]]
        if not z:
            continue
        s_here = z[0]
        field_name = text[linefeeds[i - 1] + 1 : s_here]
        field_value_start = s_here + len(sep)

        later = [s for s in separators if s > s_here]
        if later:
            # End at the last line feed before the next field's separator.
            prior_lfs = [lf for lf in linefeeds if lf < later[0]]
            field_value_end = prior_lfs[-1]
        else:
            field_value_end = linefeeds[-1]

        field_value_string = text[field_value_start:field_value_end]
        mystruct[field_name] = _coerce(field_value_string)

    return mystruct
