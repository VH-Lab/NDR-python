"""Axon ABF reader class.

Port of +ndr/+reader/axon_abf.m
Reads Axon Binary Format (.abf) files using pyabf when available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ndr.reader.base import Base


class AxonABF(Base):
    """Reader for Axon ABF (.abf) file format.

    Port of ndr.reader.axon_abf.
    """

    def __init__(self) -> None:
        super().__init__()

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Read data from specified channels."""
        try:
            import pyabf
        except ImportError as err:
            raise ImportError(
                "pyabf is required for reading ABF files. Install with: pip install pyabf"
            ) from err

        if isinstance(channel, int):
            channel = [channel]

        abf_file = self._filenamefromepochfiles(epochstreams)
        abf = pyabf.ABF(abf_file)

        if channeltype in ("time", "timestamp", "t"):
            abf.setSweep(sweepNumber=epoch_select - 1, channel=0)
            t = abf.sweepX
            return t[s0 - 1 : s1].reshape(-1, 1)

        data_list = []
        for ch in channel:
            abf.setSweep(sweepNumber=epoch_select - 1, channel=ch - 1)
            data_list.append(abf.sweepY[s0 - 1 : s1])

        return np.column_stack(data_list)

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read events (returns empty for ABF)."""
        return np.array([]), np.array([])

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List channels available for a given epoch."""
        try:
            import pyabf
        except ImportError as err:
            raise ImportError("pyabf is required for reading ABF files.") from err

        abf_file = self._filenamefromepochfiles(epochstreams)
        abf = pyabf.ABF(abf_file)

        channels: list[dict[str, Any]] = []
        channels.append({"name": "t1", "type": "time", "time_channel": 1})

        for i in range(abf.channelCount):
            channels.append({"name": f"ai{i + 1}", "type": "analog_in", "time_channel": 1})

        return channels

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> float | np.ndarray:
        """Get the sample rate."""
        try:
            import pyabf
        except ImportError as err:
            raise ImportError("pyabf is required for reading ABF files.") from err

        abf_file = self._filenamefromepochfiles(epochstreams)
        abf = pyabf.ABF(abf_file)
        return float(abf.dataRate)

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end times for an epoch."""
        try:
            import pyabf
        except ImportError as err:
            raise ImportError("pyabf is required for reading ABF files.") from err

        abf_file = self._filenamefromepochfiles(epochstreams)
        abf = pyabf.ABF(abf_file)
        abf.setSweep(sweepNumber=epoch_select - 1)
        t0 = abf.sweepX[0]
        t1 = abf.sweepX[-1]
        return [[float(t0), float(t1)]]

    @staticmethod
    def _filenamefromepochfiles(filename_array: list[str]) -> str:
        """Find the .abf file from epoch files."""
        abf_files = [f for f in filename_array if f.lower().endswith(".abf")]
        if len(abf_files) != 1:
            raise ValueError("Need exactly 1 .abf file per epoch.")
        return abf_files[0]
