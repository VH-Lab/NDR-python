"""Copy/trim SpikeGadgets .rec files.

Port of +ndr/+format/+spikegadgets/copy_rec_files.m
"""

from __future__ import annotations

from pathlib import Path

from ndr.format.spikegadgets.read_rec_config import read_rec_config


def copy_rec_files(
    filename_in: str | Path,
    s0: int,
    s1: int,
    filename_out: str | Path,
) -> None:
    """Copy a range of samples from a ``.rec`` file to a new file.

    Parameters
    ----------
    filename_in : str or Path
        Source ``.rec`` file.
    s0 : int
        First sample to copy (1-based).
    s1 : int
        Last sample to copy (1-based).
    filename_out : str or Path
        Destination file path.
    """
    filename_in = Path(filename_in)
    filename_out = Path(filename_out)

    config, _ = read_rec_config(filename_in)

    config_size = len(config["configText"])
    header_size_bytes = int(config["headerSize"]) * 2
    channel_size_bytes = int(config["numChannels"]) * 2
    block_size_bytes = header_size_bytes + 2 + channel_size_bytes

    with open(filename_in, "rb") as fin, open(filename_out, "wb") as fout:
        # Write config + first header + timestamp
        initial = fin.read(config_size + header_size_bytes + 4)
        fout.write(initial)

        # Seek to s0
        fin.seek((s0 - 1) * block_size_bytes, 1)

        # Read and write samples
        num_bytes = block_size_bytes * (s1 - s0 + 1) * 2
        data = fin.read(num_bytes)
        fout.write(data)
