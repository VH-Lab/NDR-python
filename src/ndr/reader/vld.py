"""VH Lab LabView (VHLV) reader class.

Port of +ndr/+reader/vld.m
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.vld.readvhlvdatafile import readvhlvdatafile
from ndr.format.vld.readvhlvheaderfile import readvhlvheaderfile
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType
from ndr.time.fun.times2samples import matlab_round

# MATLAB precision name -> (ndr datatype, size in bits, maxint)
_PRECISION2DATATYPE = {
    "double": ("float64", 64, 1),
    "single": ("float32", 32, 1),
    "int32": ("int32", 32, 2**31 - 1),
    "int16": ("int16", 16, 2**15 - 1),
}


class ndr_reader_vld(ndr_reader_base):
    """Reader for the VH Lab LabView (.vld/.vlh) file format.

    Port of ndr.reader.vld. An epoch is a ``.vld`` binary data file paired
    with a ``.vlh`` text header of the same base name.
    """

    def __init__(self) -> None:
        super().__init__()

    def filenamefromepochfiles(self, epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the ``.vld`` data file for the epoch."""
        pattern = re.compile(r".*\.vld$", re.IGNORECASE)
        matches = [f for f in epochstreams if pattern.match(str(f))]

        if not matches:
            raise ValueError('No file ending with ".vld" found in the provided list for the epoch.')
        if len(matches) < epoch_select:
            raise ValueError(
                f'There are only {len(matches)} ".vld" files found in the provided '
                f"list. epoch_select cannot be {epoch_select}."
            )
        return matches[epoch_select - 1]

    def readheader(self, epochstreams: list[str], epoch_select: int = 1) -> dict[str, Any]:
        """Read the ``.vlh`` header that accompanies the epoch's ``.vld`` file."""
        filename = Path(self.filenamefromepochfiles(epochstreams, epoch_select))
        return readvhlvheaderfile(filename.with_suffix(".vlh"))

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return the clock type for the epoch."""
        return [ClockType("dev_local_time")]

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end times of the epoch."""
        filename = self.filenamefromepochfiles(epochstreams, epoch_select)
        header = self.readheader(epochstreams, epoch_select)
        tot_sam = ndr_reader_vld._total_samples(header, filename)
        return [[0.0, (tot_sam - 1) / float(header["SamplingRate"])]]

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List the channels available in the epoch.

        Channels are named ``ai1``..``aiN`` by 1-based position within the
        recorded set, matching this reader's ``'indexed'`` labeling convention.
        """
        header = self.readheader(epochstreams, epoch_select)
        channels: list[dict[str, Any]] = [{"name": "t1", "type": "time", "time_channel": 1}]
        for i in range(1, int(header["NumChans"]) + 1):
            channels.append({"name": f"ai{i}", "type": "analog_in", "time_channel": 1})
        return channels

    def underlying_datatype(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        channeltype: str = "analog_in",
        channel: int | list[int] = 1,
    ) -> tuple[str, np.ndarray, int]:
        """Return the underlying storage datatype, scaling polynomial, and bit size."""
        if isinstance(channel, (int, np.integer)):
            channel = [int(channel)]
        n = len(channel)

        if channeltype.lower() in ("time", "t"):
            return "float64", np.tile(np.array([0.0, 1.0]), (n, 1)), 64

        if channeltype.lower() in ("analog_in", "ai"):
            header = self.readheader(epochstreams, epoch_select)
            datatype, datasize, maxint = ndr_reader_vld._precision2datatype(header)
            scale = float(header["Scale"]) / maxint if "Scale" in header else 1.0
            return datatype, np.tile(np.array([0.0, scale]), (n, 1)), datasize

        warnings.warn(
            f'Unknown channel type "{channeltype}" requested for '
            "underlying_datatype. Using base class default.",
            stacklevel=2,
        )
        return super().underlying_datatype(epochstreams, epoch_select, channeltype, channel)

    def readchannels_epochsamples(
        self,
        channeltype: str | list[str],
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int = 1,
        s0: int | float = -np.inf,
        s1: int | float = np.inf,
    ) -> np.ndarray:
        """Read data from the specified channels between samples ``s0`` and ``s1``.

        ``channeltype`` may be a single string or a uniform list of identical
        channel-type strings; this reader reads one type per call.
        """
        if isinstance(channeltype, list):
            if not all(ct == channeltype[0] for ct in channeltype):
                raise ValueError(
                    "channeltype list must be uniform; the vld reader reads one type per call."
                )
            channeltype = channeltype[0]

        if isinstance(channel, (int, np.integer)):
            channel = [int(channel)]

        filename = self.filenamefromepochfiles(epochstreams, epoch_select)
        header = self.readheader(epochstreams, epoch_select)
        sr = float(header["SamplingRate"])
        tot_sam = ndr_reader_vld._total_samples(header, filename)

        if np.isinf(s0):
            s0 = 1
        elif s0 < 1:
            warnings.warn(
                "Starting sample number must be a positive integer. Using default value (s0 = 1).",
                stacklevel=2,
            )
            s0 = 1
        elif s0 != matlab_round(s0):
            s0 = int(matlab_round(s0))
            warnings.warn(
                "Starting sample number must be an integer. Using closest "
                f"integer value (s0 = {s0}).",
                stacklevel=2,
            )

        if np.isinf(s1):
            s1 = tot_sam
        elif s1 < 1:
            raise ValueError("Ending sample number must be a positive integer.")
        elif s1 != matlab_round(s1):
            s1 = int(matlab_round(s1))
            warnings.warn(
                "Ending sample number must be an integer. Using closest "
                f"integer value (s1 = {s1}).",
                stacklevel=2,
            )

        if s1 > tot_sam:
            warnings.warn(
                "Ending sample number is greater than the length of the data. "
                f"Using last sample (s1 = {int(tot_sam)}).",
                stacklevel=2,
            )
            s1 = tot_sam

        s0 = int(s0)
        s1 = int(s1)
        t0 = (s0 - 1) / sr
        t1 = (s1 - 1) / sr

        if channeltype.lower() in ("time", "t"):
            T, _D, _tot_sam, _tot_time = readvhlvdatafile(filename, header, 1, t0, t1)
            return np.tile(np.asarray(T).reshape(-1, 1), (1, len(channel)))

        _T, D, _tot_sam, _tot_time = readvhlvdatafile(filename, header, channel, t0, t1)
        return D

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        channeltype: str = "analog_in",
        channel: int | list[int] = 1,
    ) -> float | np.ndarray:
        """Return the sample rate for the given channels.

        Every channel in a VHLV recording shares one sampling rate.
        """
        header = self.readheader(epochstreams, epoch_select)
        sr = float(header["SamplingRate"])
        if isinstance(channel, (int, np.integer)):
            return sr
        return np.full(len(channel), sr)

    def daqchannels2internalchannels(
        self,
        channelprefix: list[str],
        channelnumber: list[int] | np.ndarray,
        epochstreams: list[str],
        epoch_select: int = 1,
    ) -> list[dict[str, Any]]:
        """Map DAQ channel prefixes/numbers onto this reader's internal channels."""
        from ndr.string.channelstring2channels import channelstring2channels

        channels_available = self.getchannelsepoch(epochstreams, epoch_select)
        channelstruct: list[dict[str, Any]] = []

        if len(channelprefix) != len(channelnumber):
            raise ValueError("Number of channel prefixes must match number of channel numbers.")

        for prefix, number in zip(channelprefix, channelnumber):
            current_prefix = prefix.lower()
            current_number = int(number)
            found = False

            for avail in channels_available:
                avail_prefix, avail_number = channelstring2channels(avail["name"])
                if (
                    avail_prefix
                    and avail_number
                    and current_prefix == avail_prefix[0].lower()
                    and current_number == avail_number[0]
                ):
                    channelstruct.append(
                        {
                            "internal_type": avail["type"],
                            "internal_number": current_number,
                            "internal_channelname": avail["name"],
                            "ndr_type": ndr_reader_base.mfdaq_type(avail["type"]),
                            "samplerate": self.samplerate(
                                epochstreams, epoch_select, current_prefix, current_number
                            ),
                        }
                    )
                    found = True
                    break

            if not found:
                warnings.warn(
                    f"Requested channel {current_prefix}{current_number} not found in epoch.",
                    stacklevel=2,
                )

        return channelstruct

    def readevents_epochsamples_native(
        self,
        channeltype: str | list[str],
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """VHLV recordings carry no native event channels."""
        return np.array([]), np.array([])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _precision2datatype(header: dict[str, Any]) -> tuple[str, int, int]:
        """Return (datatype, datasize_bits, maxint) for the header's precision."""
        precision = header.get("precision", "double")
        if precision not in _PRECISION2DATATYPE:
            raise ValueError(f"Unknown precision {precision} in VHLV header.")
        return _PRECISION2DATATYPE[precision]

    @staticmethod
    def _total_samples(header: dict[str, Any], filename: str | Path) -> float:
        """Estimate the total number of samples per channel from the file size."""
        _datatype, datasize, _maxint = ndr_reader_vld._precision2datatype(header)
        unit_size = datasize // 8
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Could not find file {filename} to determine its size.")
        return path.stat().st_size / (int(header["NumChans"]) * unit_size)
