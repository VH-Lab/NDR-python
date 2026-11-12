"""Describe SmartSPIM's stitched ``All_Channels`` volume.

Port of ``+ndr/+format/+smartspim/readAllChannelsInfo.m``.

``rootDir/All_Channels`` holds the output of the SmartSPIM AutoStitch
pipeline: a TeraStitcher merge of the destriped ``Ex_*`` tiles with the
per-channel ``<wavelength>_flat.tiff`` flat-field / left-right
illumination correction applied and the tile overlaps blended. It is the
volume Imaris is built from, and the one analysis should use
(ferret-lightsheet ``formats/SmartSPIM.md``, sections 4, 6 and 7).

``All_Channels`` is a flat folder of 2-D TIFFs, one per (z, channel),
named ``<Y origin>_<X origin>_<z*10>_ch<k>.tiff`` (e.g.
``675050_455000_000110_ch1.tiff``) or, by other versions of the
pipeline, ``<z*10>_ch<k>.tiff`` (e.g. ``000110_ch1.tiff``) or
``Z<z*10>_Ch<k>.tif`` (e.g. ``Z000110_Ch0.tif``; case does not matter),
plus an ``ImageList.txt`` for Imaris. Every slice spans the whole stitched canvas (height = Y,
width = X), so there are no tiles. The ``ch<k>`` index follows the
filter-channel number, NOT the acquisition order. It is taken from the
``<channel>_MIP_stitched`` folder, whose files carry the same ``_ch<k>``
suffix; when that folder is absent (e.g. a partial restore), ``k`` is
the rank of the channel's filter-channel number among all channels
(Ch1 -> ch0, Ch2 -> ch1, ...). Between the two, ``IMS_Notes.txt`` ("The
folder <channel>_stitched is represented as Ch<k>", written when SmartSPIM
converts All_Channels to Imaris; :func:`readImsNotes`) is used when it names
the channel; SmartSPIM 6.0 writes no ``_ch`` suffix in ``_MIP_stitched``. A
channel folder without ``_Ch<n>`` (``Ex_<laser>_Em_<filter>``) gets its
filter-channel number from metadata.json (see :func:`listChannels`); a single
one takes whichever one ``_ch<k>`` the slices carry. When such folders are
mapped by filter-channel rank, that rank must match the order the channels
first appear in metadata.json's tiles, or :class:`ValueError` is raised
(MATLAB ``ndr:format:smartspim:readAllChannelsInfo:ChannelOrderDisagrees``):
which order the stitcher follows for them is not yet known.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import validate_call

from ndr.format.smartspim.listChannels import listChannels
from ndr.format.smartspim.readAcquisitionMetadata import readAcquisitionMetadata
from ndr.format.smartspim.readImsNotes import readImsNotes

__all__ = ["readAllChannelsInfo"]

# e.g. 675050_455000_000110_ch0.tiff -> (z*10, k), and, as other versions
# of the pipeline name them, without the origins: 000110_ch0.tiff
_SLICE_RE = re.compile(r"^(?:\d+_\d+_|Z)?(\d+)_ch(\d+)\.tiff?$", re.IGNORECASE)

_MATLAB_CLASS = {"float32": "single", "float64": "double", "bool": "logical"}


def _tifffile():
    """Import tifffile lazily so it is only required when a TIFF is read."""
    try:
        import tifffile
    except ImportError as err:  # pragma: no cover - exercised only without tifffile
        raise ImportError(
            "tifffile is required for reading SmartSPIM All_Channels slices. "
            'Install with: pip install "ndr[formats]"'
        ) from err
    return tifffile


def _index_slices(folder: Path) -> list[tuple[int, int, str]]:
    """Return ``(k, z, path)`` for every ``*_ch<k>.tif(f)`` file in ``folder``."""
    out = []
    for e in folder.iterdir():
        if e.is_dir():
            continue
        m = _SLICE_RE.match(e.name)
        if m:
            out.append((int(m.group(2)), int(m.group(1)), str(folder / e.name)))
    # A folder cannot hold one name twice, but a network share's listing can
    # return it twice (and leave others out): macOS SMB directory caching
    # does, on large folders. A volume built from it would be wrong.
    paths = [p for _, _, p in out]
    if len(set(paths)) < len(paths):
        raise RuntimeError(
            f"The listing of {folder} names some files more than once, so it "
            "cannot be trusted (on a macOS SMB share, turn off directory "
            "caching: dir_cache_off=yes in /etc/nsmb.conf, then remount)."
        )
    keys = [(k, z) for k, z, _ in out]
    if len(set(keys)) < len(keys):
        raise RuntimeError(
            f"{folder} holds more than one file for the same channel and z "
            "(tiles rather than stitched slices?)."
        )
    return out


def _check_rank_matches_metadata_order(rootDir: str, acq_channels: list[dict[str, Any]]) -> None:
    """Refuse when filter-channel order and metadata.json order disagree.

    Channel folders without ``_Ch<n>`` (SmartSPIM 6.0) and no direct record
    of which ``_Ch<k>`` is whose: the filter-channel rank decides, as for
    numbered folders. Until an acquisition shows which of the two the
    stitcher follows there, refuse when that rank and the order the channels
    first appear in metadata.json's tiles disagree.
    """
    try:
        tiles = list(readAcquisitionMetadata(rootDir)["tiles"])
    except (OSError, ValueError):
        tiles = []
    first_seen = []
    for c in acq_channels:
        j = next(
            (
                i
                for i, t in enumerate(tiles)
                if t["laser"] == c["wavelength"] and t["filter"] == c["filterName"]
            ),
            math.inf,
        )
        first_seen.append(j)
    by_filter = sorted(range(len(acq_channels)), key=lambda i: acq_channels[i]["filterChannel"])
    by_metadata = sorted(range(len(acq_channels)), key=lambda i: first_seen[i])
    if math.inf in first_seen or by_filter != by_metadata:
        raise ValueError(
            f'In {rootDir} the channel folders have no "_Ch<n>" and nothing records which '
            'All_Channels "_ch<k>" is which channel, and the channels\' filter-channel order '
            f"({', '.join(acq_channels[i]['name'] for i in by_filter)}) differs from the order "
            f"they appear in metadata.json ({', '.join(acq_channels[i]['name'] for i in by_metadata)}), "
            "so it is not known which order the stitcher used."
        )


def _channel_index(
    root: Path,
    channel_name: str,
    acq_channels: list[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> tuple[int, bool]:
    """The channel's _ch<k>, and whether it came from the filter-channel rank."""
    # Preferred: the single _ch<k> suffix used in <channel>_MIP_stitched.
    mip_dir = root / f"{channel_name}_MIP_stitched"
    if mip_dir.is_dir():
        ks = {k for k, _, _ in _index_slices(mip_dir)}
        if len(ks) == 1:
            return ks.pop(), False
    # Next: SmartSPIM's own statement in IMS_Notes.txt ("The folder
    # <channel>_stitched is represented as Ch<k>").
    for n in notes:
        if n["channelName"] == channel_name:
            return int(n["chIndex"]), False
    # Fallback: rank of the filter-channel number (stable on ties).
    ranked = sorted(acq_channels, key=lambda c: c["filterChannel"])
    return [c["name"] for c in ranked].index(channel_name), True


@validate_call
def readAllChannelsInfo(rootDir: str, *, ReadHeader: bool = True) -> dict[str, Any]:
    """Return a dict describing the stitched ``All_Channels`` volume.

    Fields:

    - ``allChannelsDir`` (str): path of ``rootDir/All_Channels``
    - ``channels`` (list[dict]): one entry per channel, sorted by
      ``chIndex``, with ``name`` (channel directory name), ``chIndex``
      (float, the ``k`` of its ``_ch<k>`` files), ``numSlices`` (float)
      and ``slicePaths`` (list[str] sorted by the numeric z field)
    - ``channelNames`` (list[str]): channel names in ``chIndex`` order
    - ``height`` / ``width`` (float): slice size in pixels, NaN if not read
    - ``dtype`` (str): MATLAB numeric class of the pixels, ``''`` if not read
    - ``voxelSizeUm`` (list[float]): ``[z, y, x]`` in micrometers from
      ``metadata.json``, NaNs if unavailable

    With ``ReadHeader=False`` the first slice's TIFF header is not opened
    and ``height``/``width``/``dtype`` come back NaN/NaN/``''``; this is
    what :func:`hasAllChannels` uses for its cheap probe.

    Raises :class:`NotADirectoryError` if ``rootDir`` is not a folder,
    :class:`FileNotFoundError` if ``All_Channels`` is missing, holds no
    ``*_ch<k>.tif(f)`` slices or lacks the slices of a channel, and
    :class:`ValueError` if two channels map to the same ``ch<k>`` or the
    first slice cannot be read.
    """
    if not rootDir:
        raise ValueError("rootDir must be a non-empty string.")
    root = Path(rootDir)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {rootDir}")

    all_dir = root / "All_Channels"
    if not all_dir.is_dir():
        raise FileNotFoundError(f"All_Channels directory not found: {all_dir}")

    index = _index_slices(all_dir)
    if not index:
        raise FileNotFoundError(f'No "*_ch<k>.tif(f)" slices found in {all_dir}')

    acq_channels = listChannels(rootDir)
    names = [c["name"] for c in acq_channels]
    ks = {kk for kk, _, _ in index}
    if (
        len(acq_channels) == 1
        and not re.search(r"_Ch\d+$", acq_channels[0]["name"])
        and len(ks) == 1
    ):
        # A single channel folder without "_Ch<n>": its slices are the only
        # _ch<k> there is, whatever number metadata.json gives it.
        ch_indices = [next(iter(ks))]
    else:
        notes = readImsNotes(rootDir)
        found = [_channel_index(root, n, acq_channels, notes) for n in names]
        ch_indices = [k for k, _ in found]
        unnumbered = any(not re.search(r"_Ch\d+$", n) for n in names)
        if len(names) > 1 and unnumbered and any(by_rank for _, by_rank in found):
            _check_rank_matches_metadata_order(rootDir, acq_channels)
    if len(set(ch_indices)) != len(ch_indices):
        raise ValueError(f"Two channels map to the same All_Channels _ch<k> index in {rootDir}.")

    order = sorted(range(len(names)), key=lambda i: ch_indices[i])
    channels = []
    for i in order:
        k = ch_indices[i]
        sel = sorted((z, p) for kk, z, p in index if kk == k)
        if not sel:
            raise FileNotFoundError(f'No "_ch{k}" slices for {names[i]} in {all_dir}')
        paths = [p for _, p in sel]
        channels.append(
            {
                "name": names[i],
                "chIndex": float(k),
                "numSlices": float(len(paths)),
                "slicePaths": paths,
            }
        )

    height = math.nan
    width = math.nan
    dtype = ""
    if ReadHeader:
        first_path = channels[0]["slicePaths"][0]
        tifffile = _tifffile()
        try:
            with tifffile.TiffFile(first_path) as tf:
                page = tf.pages[0]
                height = float(page.imagelength)
                width = float(page.imagewidth)
                np_dtype = np.dtype(page.dtype)
        except Exception as err:
            raise ValueError(f"Failed to read TIFF header of {first_path}: {err}") from err
        dtype = _MATLAB_CLASS.get(np_dtype.name, np_dtype.name)

    voxel_size_um = [math.nan, math.nan, math.nan]
    try:
        voxel_size_um = [float(v) for v in readAcquisitionMetadata(rootDir)["voxelSizeUm"]]
    except Exception:
        # metadata.json missing or unreadable; leave voxel size unknown.
        pass

    return {
        "allChannelsDir": str(all_dir),
        "channels": channels,
        "channelNames": [c["name"] for c in channels],
        "height": height,
        "width": width,
        "dtype": dtype,
        "voxelSizeUm": voxel_size_um,
    }
