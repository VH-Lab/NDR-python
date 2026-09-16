"""Read Intan "one file per signal type" data.

Port of +ndr/+format/+intan/read_IntanRHD2000_one_file_per_channel_type.m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_FILENAME_BY_CHANNEL_TYPE = {
    1: "time.dat",
    2: "amplifier.dat",
    3: "auxiliary.dat",
    4: "supply.dat",
    # 5: temperature -- Intan does not write a standalone file for this in
    # the "one file per signal type" format; temperature is interleaved
    # with the supply-voltage stream. Kept out of the map on purpose so a
    # caller that requests it hits the ValueError below.
    6: "analogin.dat",
    7: "digitalin.dat",
    8: "digitalout.dat",
}

_SAMPLE_SIZE_BYTES = {1: 4, 2: 2, 3: 2, 4: 2, 6: 2, 7: 2, 8: 2}
_SAMPLE_DTYPE = {
    1: np.int32,
    2: np.int16,
    3: np.uint16,
    4: np.uint16,
    6: np.uint16,
    7: np.uint16,
    8: np.uint16,
}


def read_IntanRHD2000_one_file_per_channel_type(
    directory_name: str | Path,
    channel_type: int,
    channel_index: int,
    num_channels: int,
    s0: int,
    s1: int,
) -> np.ndarray:
    """Read one channel's samples from the Intan "one file per signal type" layout.

    Parameters
    ----------
    directory_name : str or Path
        Directory containing the ``.dat`` files. Files may carry an Intan
        timestamp prefix or a lab-specific prefix
        (e.g. ``febc0_u000_000_amplifier.dat``); the reader resolves the
        prefix automatically.
    channel_type : int
        Signal type: ``1`` time.dat, ``2`` amplifier.dat, ``3`` auxiliary.dat,
        ``4`` supply.dat, ``6`` analogin.dat, ``7`` digitalin.dat,
        ``8`` digitalout.dat. Temperature (5) is not saved as a standalone
        file in this layout.
    channel_index : int
        1-based position of the channel within the enabled channels of this
        type. Ignored for digital channels (the caller extracts the bit).
    num_channels : int
        Total number of enabled channels of this type.
    s0, s1 : int
        1-based inclusive sample range.

    Returns
    -------
    numpy.ndarray
        For analog channels, a 1-D array of the requested samples in the
        file's native dtype. For digital channels, the full packed 16-bit
        word per sample; the caller extracts the requested bit.

    Filenames match the Intan-RHX / IntanToNWB / python-neo spec.

    See also: ``read_Intan_RHD2000_directory``.
    """
    directory_name = Path(directory_name)
    if not directory_name.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {directory_name}")
    if channel_type not in _FILENAME_BY_CHANNEL_TYPE:
        raise ValueError(f"Unknown channel_type {channel_type}")
    if channel_index < 1 or num_channels < 1:
        raise ValueError("channel_index and num_channels must be positive integers")
    if s0 < 1 or s1 < s0:
        raise ValueError("s0 must be >= 1 and s1 must be >= s0")

    filename_post = _FILENAME_BY_CHANNEL_TYPE[channel_type]
    fn = _fixdatfilename(directory_name / filename_post)
    if fn is None:
        raise FileNotFoundError(
            f"Could not find data file matching *{filename_post} in {directory_name}."
        )

    sample_size = _SAMPLE_SIZE_BYTES[channel_type]
    dtype = np.dtype(_SAMPLE_DTYPE[channel_type]).newbyteorder("<")
    count = s1 - s0 + 1

    with open(fn, "rb") as fid:
        if channel_type < 7:
            # Interleaved: seek past (s0-1) full sample groups, then past
            # (channel_index-1) samples within the current group, and stride
            # over the other channels on each subsequent read.
            fid.seek(sample_size * (num_channels * (s0 - 1) + (channel_index - 1)))
            if num_channels == 1:
                return np.fromfile(fid, dtype=dtype, count=count)
            # Read count samples with a stride of num_channels. numpy has no
            # strided fread, so read the whole run and slice.
            span = num_channels * (count - 1) + 1
            raw = np.fromfile(fid, dtype=dtype, count=span)
            return raw[::num_channels]
        # Digital: one packed 16-bit word per sample; caller extracts bit.
        fid.seek(sample_size * (s0 - 1))
        return np.fromfile(fid, dtype=dtype, count=count)


def _fixdatfilename(filename: Path) -> Path | None:
    """Resolve a filename that may carry an Intan timestamp or lab prefix.

    Returns the exact path if it exists, otherwise the first sibling matching
    ``*<basename>`` in the same directory, otherwise ``None``.
    """
    if filename.is_file():
        return filename
    parent = filename.parent
    stem_ext = filename.name
    matches = sorted(parent.glob(f"*{stem_ext}"))
    if matches:
        return matches[0]
    return None
