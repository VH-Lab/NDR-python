"""Copy a range of data blocks from one RHD file to another.

Port of +ndr/+format/+intan/copy_Intan_RHD2000_blocks.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header
from ndr.format.intan.read_Intan_RHD2000_datafile import (
    Intan_RHD2000_blockinfo,
    _get_header_size,
)


def copy_Intan_RHD2000_blocks(
    filename_in: str | Path,
    b1: int,
    b2: int,
    filename_out: str | Path,
) -> int:
    """Copy a range of data blocks from an RHD file to a new file.

    Parameters
    ----------
    filename_in : str or Path
        Source RHD file.
    b1 : int
        First block to copy (1-based).
    b2 : int
        Last block to copy (1-based, inclusive).
    filename_out : str or Path
        Destination file path.

    Returns
    -------
    int
        Status code (0 on success).

    Raises
    ------
    ValueError
        If the requested block range is out of bounds.
    """
    chunk_size = 100  # read at most 100 blocks per step

    filename_in = Path(filename_in)
    filename_out = Path(filename_out)

    header = read_Intan_RHD2000_header(filename_in)
    blockinfo, bytes_per_block, bytes_present, num_data_blocks = Intan_RHD2000_blockinfo(
        filename_in, header
    )
    header_size = blockinfo["header_size"]

    if b1 < 1 or b2 > num_data_blocks:
        raise ValueError(
            f"Requested block out of the range 1..{num_data_blocks}."
        )

    # Build list of block-range chunks
    block_starts = list(range(b1, b2 + 1, chunk_size))
    block_ends = [min(bs + chunk_size - 1, b2) for bs in block_starts]

    with open(filename_in, "rb") as fin, open(filename_out, "wb") as fout:
        # Write header
        fin.seek(0)
        fout.write(fin.read(header_size))

        # Copy requested blocks in chunks
        for bs, be in zip(block_starts, block_ends):
            fin.seek(header_size + (bs - 1) * bytes_per_block)
            num_blocks = be - bs + 1
            data = fin.read(num_blocks * bytes_per_block)
            fout.write(data)

    return 0
