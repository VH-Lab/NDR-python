"""VHLAB LabView (.vld/.vlh) reader class.

Port of +ndr/+reader/vld.m

Reads data from the VHLAB LabView multichannel acquisition system binary
format. Each recording epoch is described by a text header file (``.vlh``) and
a big-endian binary data file (``.vld``). All acquired channels are analog
input channels (``ai1``, ``ai2``, ..., numbered in acquisition order) that
share a single sampling rate and a single time channel (``t1``).

See also: ndr.reader.base, ndr.format.vld.readvhlvheaderfile,
ndr.format.vld.readvhlvdatafile.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from ndr.format.vld.readvhlvdatafile import readvhlvdatafile, total_samples
from ndr.format.vld.readvhlvheaderfile import readvhlvheaderfile
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType


class ndr_reader_vld(ndr_reader_base):
    """Reader for the VHLAB LabView (.vld/.vlh) file format.

    Port of ndr.reader.vld.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # File / header resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _filenamefromepochfiles(epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the ``.vld`` data filename from a list of epoch files.

        Port of ndr.reader.vld.filenamefromepochfiles.
        """
        pattern = re.compile(r".*\.vld$", re.IGNORECASE)
        vld_files = [f for f in epochstreams if pattern.match(f)]
        if len(vld_files) == 0:
            raise ValueError('No file ending with ".vld" found in the provided list for the epoch.')
        if len(vld_files) < epoch_select:
            raise ValueError(
                f'There are only {len(vld_files)} ".vld" files found in the '
                f"provided list. epoch_select cannot be {epoch_select}."
            )
        return vld_files[epoch_select - 1]

    def readheader(self, epochstreams: list[str], epoch_select: int = 1) -> dict[str, Any]:
        """Read the VHLV header structure for an epoch.

        Locates the ``.vld`` data file, derives the matching ``.vlh`` header
        file (same path and base name), and returns the parsed header.

        Port of ndr.reader.vld.readheader.
        """
        filename = self._filenamefromepochfiles(epochstreams, epoch_select)
        base, _ = os.path.splitext(filename)
        headerfile = base + ".vlh"
        return readvhlvheaderfile(headerfile)

    # ------------------------------------------------------------------
    # Clock / timing
    # ------------------------------------------------------------------

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return the clock types available for this epoch.

        VHLV files record time relative to the beginning of the recording, so a
        single 'dev_local_time' clock is returned.
        """
        return [ClockType("dev_local_time")]

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end epoch times.

        Sample 1 occurs at ``t==0``, so ``t0`` is 0 and ``t1`` is
        ``(total_samples-1)/SamplingRate``.
        """
        filename = self._filenamefromepochfiles(epochstreams, epoch_select)
        header = self.readheader(epochstreams, epoch_select)
        tot_sam = total_samples(header, filename)
        t0 = 0.0
        t1 = (tot_sam - 1) / float(header["SamplingRate"])
        return [[t0, t1]]

    # ------------------------------------------------------------------
    # Channels
    # ------------------------------------------------------------------

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List the channels available for a given epoch.

        VHLV files store one or more analog input channels (``ai1``, ``ai2``,
        ...) that share a single time channel (``t1``). The time channel is
        listed first, followed by ``NumChans`` analog inputs.

        Each returned dict has keys ``name``, ``type``, ``time_channel``.
        """
        header = self.readheader(epochstreams, epoch_select)
        channels: list[dict[str, Any]] = [{"name": "t1", "type": "time", "time_channel": 1}]
        for i in range(1, int(header["NumChans"]) + 1):
            channels.append({"name": f"ai{i}", "type": "analog_in", "time_channel": 1})
        return channels

    def underlying_datatype(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> tuple[str, np.ndarray, int]:
        """Get the native stored data type for channels.

        For analog channels, the stored precision is read from the header; when
        a ``Scale`` field is present the integers are scaled by ``Scale/maxint``
        and this is reflected in the polynomial ``p = [offset scale]``. Time is
        ``float64``.
        """
        if isinstance(channel, int):
            n_channels = 1
        else:
            n_channels = len(channel)

        ct = channeltype.lower()
        if ct in ("time", "t"):
            datatype = "float64"
            datasize = 64
            p = np.tile([0.0, 1.0], (n_channels, 1))
        elif ct in ("analog_in", "ai"):
            header = self.readheader(epochstreams, epoch_select)
            datatype, datasize, maxint = self._precision2datatype(header)
            if "Scale" in header:
                scale = float(header["Scale"]) / maxint
            else:
                scale = 1.0
            p = np.tile([0.0, scale], (n_channels, 1))
        else:
            return super().underlying_datatype(epochstreams, epoch_select, channeltype, channel)

        return datatype, p, datasize

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Read data samples for the specified channels between s0 and s1.

        Samples are 1-based and inclusive; sample 1 occurs at ``t==0``. For a
        ``time`` channel, the returned data are the timestamps (seconds
        relative to the start of the recording) of each sample.

        Port of ndr.reader.vld.readchannels_epochsamples.
        """
        if isinstance(channeltype, (list, tuple)):
            first = channeltype[0]
            if not all(ct == first for ct in channeltype):
                raise ValueError(
                    "channeltype cell array must be uniform; the vld reader "
                    "reads one type per call."
                )
            channeltype = first

        if isinstance(channel, int):
            channel = [channel]

        filename = self._filenamefromepochfiles(epochstreams, epoch_select)
        header = self.readheader(epochstreams, epoch_select)
        sr = float(header["SamplingRate"])
        tot_sam = total_samples(header, filename)

        # Resolve sample bounds; sample 1 occurs at t==0
        s0 = int(round(s0))
        s1 = int(round(s1))
        if s0 < 1:
            s0 = 1
        if s1 < 1:
            raise ValueError("Ending sample number must be a positive integer.")
        if s1 > tot_sam:
            s1 = tot_sam

        t0 = (s0 - 1) / sr
        t1 = (s1 - 1) / sr

        if channeltype.lower() in ("time", "t"):
            T, _, _, _ = readvhlvdatafile(filename, header, 1, t0, t1)
            T = np.asarray(T).reshape(-1, 1)
            return np.tile(T, (1, len(channel)))
        else:
            _, D, _, _ = readvhlvdatafile(filename, header, channel, t0, t1)
            return np.asarray(D)

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Not applicable: VHLV files contain only regularly-sampled analog
        data, so there are no native event/marker channels. Returns empty
        arrays."""
        return np.array([]), np.array([])

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> float | np.ndarray:
        """Get the sample rate for the specified channels.

        For VHLV files the sampling rate is constant across all channels and is
        read from the header.
        """
        header = self.readheader(epochstreams, epoch_select)
        sr = float(header["SamplingRate"])
        if isinstance(channel, int):
            return sr
        return np.full(len(channel), sr, dtype=float)

    # ------------------------------------------------------------------
    # Static helpers (private in MATLAB)
    # ------------------------------------------------------------------

    @staticmethod
    def _precision2datatype(header: dict[str, Any]) -> tuple[str, int, float]:
        """Map a VHLV header precision to (datatype, datasize_bits, maxint).

        Port of ndr.reader.vld.precision2datatype.
        """
        precision = header.get("precision", "double")
        table = {
            "double": ("float64", 64, 1.0),
            "single": ("float32", 32, 1.0),
            "int32": ("int32", 32, float(2**31 - 1)),
            "int16": ("int16", 16, float(2**15 - 1)),
        }
        if precision not in table:
            raise ValueError(f"Unknown precision {precision} in VHLV header.")
        return table[precision]
