"""Concatenate all RHD files in a directory and rename originals.

Port of +ndr/+format/+intan/concat_rhd_files.m
"""

from __future__ import annotations

from pathlib import Path

from ndr.format.intan.cat_Intan_RHD2000_files import cat_Intan_RHD2000_files


def concat_rhd_files(dirname: str | Path) -> None:
    """Concatenate all RHD files in a directory and rename the originals.

    All ``*.rhd`` files in *dirname* are concatenated into one larger file
    (using :func:`cat_Intan_RHD2000_files`), and the original files are
    renamed with a ``.rhd_original`` extension.

    Parameters
    ----------
    dirname : str or Path
        Path to the directory containing ``.rhd`` files.
    """
    dirname = Path(dirname)
    rhd_files = sorted(dirname.glob("*.rhd"))

    if not rhd_files:
        return

    cat_Intan_RHD2000_files(*rhd_files)

    for f in rhd_files:
        f.rename(f.with_suffix(".rhd_original"))
