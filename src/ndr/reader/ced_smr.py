"""CED SMR reader class.

Port of +ndr/+reader/ced_smr.m
Reads Cambridge Electronic Design Spike2 (.smr) files using neo.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ndr.reader.base import Base


class CedSMR(Base):
    """Reader for CED Spike2 (.smr) file format.

    Port of ndr.reader.ced_smr.
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
            from neo.io import Spike2IO
        except ImportError as err:
            raise ImportError(
                "neo is required for reading SMR files. Install with: pip install neo"
            ) from err

        if isinstance(channel, int):
            channel = [channel]

        smr_file = self._filenamefromepochfiles(epochstreams)
        reader = Spike2IO(filename=smr_file)
        block = reader.read_block(signal_group_mode="split-all")
        seg = block.segments[epoch_select - 1]

        if channeltype in ("time", "timestamp", "t"):
            sig = seg.analogsignals[0]
            times = sig.times.magnitude
            return times[s0 - 1 : s1].reshape(-1, 1)

        data_list = []
        for ch in channel:
            sig = seg.analogsignals[ch - 1]
            data_list.append(sig.magnitude[s0 - 1 : s1].flatten())

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
        """Read events from CED files."""
        try:
            from neo.io import Spike2IO
        except ImportError as err:
            raise ImportError("neo is required for reading SMR files.") from err

        smr_file = self._filenamefromepochfiles(epochstreams)
        reader = Spike2IO(filename=smr_file)
        block = reader.read_block(signal_group_mode="split-all")
        seg = block.segments[epoch_select - 1]

        timestamps_all = []
        data_all = []

        for evt in seg.events:
            times = evt.times.magnitude
            mask = (times >= t0) & (times <= t1)
            timestamps_all.append(times[mask])
            data_all.append(np.ones(np.sum(mask)))

        if timestamps_all:
            return np.concatenate(timestamps_all), np.concatenate(data_all)
        return np.array([]), np.array([])

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List channels available for a given epoch."""
        try:
            from neo.io import Spike2IO
        except ImportError as err:
            raise ImportError("neo is required for reading SMR files.") from err

        smr_file = self._filenamefromepochfiles(epochstreams)
        reader = Spike2IO(filename=smr_file)
        block = reader.read_block(signal_group_mode="split-all")
        seg = block.segments[epoch_select - 1]

        channels: list[dict[str, Any]] = []
        channels.append({"name": "t1", "type": "time", "time_channel": 1})

        for i, _sig in enumerate(seg.analogsignals):
            channels.append({"name": f"ai{i + 1}", "type": "analog_in", "time_channel": 1})

        for i, _evt in enumerate(seg.events):
            channels.append({"name": f"e{i + 1}", "type": "event", "time_channel": 1})

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
            from neo.io import Spike2IO
        except ImportError as err:
            raise ImportError("neo is required for reading SMR files.") from err

        smr_file = self._filenamefromepochfiles(epochstreams)
        reader = Spike2IO(filename=smr_file)
        block = reader.read_block(signal_group_mode="split-all")
        seg = block.segments[epoch_select - 1]

        if isinstance(channel, int):
            channel = [channel]

        sr_list = []
        for ch in channel:
            if ch - 1 < len(seg.analogsignals):
                sr_list.append(float(seg.analogsignals[ch - 1].sampling_rate.magnitude))
            else:
                sr_list.append(float("nan"))

        if len(sr_list) == 1:
            return sr_list[0]
        return np.array(sr_list)

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end times for an epoch."""
        try:
            from neo.io import Spike2IO
        except ImportError as err:
            raise ImportError("neo is required for reading SMR files.") from err

        smr_file = self._filenamefromepochfiles(epochstreams)
        reader = Spike2IO(filename=smr_file)
        block = reader.read_block(signal_group_mode="split-all")
        seg = block.segments[epoch_select - 1]

        if seg.analogsignals:
            sig = seg.analogsignals[0]
            t0 = float(sig.t_start.magnitude)
            t1 = float(sig.t_stop.magnitude)
        else:
            t0 = float("nan")
            t1 = float("nan")

        return [[t0, t1]]

    @staticmethod
    def _filenamefromepochfiles(filename_array: list[str]) -> str:
        """Find the .smr file from epoch files."""
        smr_files = [f for f in filename_array if f.lower().endswith(".smr")]
        if len(smr_files) != 1:
            raise ValueError("Need exactly 1 .smr file per epoch.")
        return smr_files[0]
