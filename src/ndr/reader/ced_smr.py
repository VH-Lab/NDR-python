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

    @staticmethod
    def _load_segment(epochstreams: list[str], epoch_select: int = 1):
        """Load a Neo segment from SMR file."""
        try:
            from neo.io import Spike2IO
        except ImportError as err:
            raise ImportError(
                "neo is required for reading SMR files. Install with: pip install neo"
            ) from err

        smr_file = CedSMR._filenamefromepochfiles(epochstreams)
        reader = Spike2IO(filename=smr_file)
        block = reader.read_block(signal_group_mode="split-all")
        return block.segments[epoch_select - 1]

    @staticmethod
    def _find_analog_by_ced_channel(seg, ced_channel: int):
        """Find a Neo analog signal by CED channel number (1-based).

        CED channel N maps to Neo ch_id N-1.
        """
        target_id = ced_channel - 1
        for sig in seg.analogsignals:
            ch_ids = sig.array_annotations.get("channel_ids", [])
            if len(ch_ids) > 0 and int(ch_ids[0]) == target_id:
                return sig
        raise ValueError(f"No analog signal found for CED channel {ced_channel}")

    @staticmethod
    def _find_event_by_ced_channel(seg, ced_channel: int):
        """Find a Neo event by CED channel number (1-based).

        CED channel N maps to Neo event id N-1.
        """
        target_id = str(ced_channel - 1)
        for evt in seg.events:
            if evt.annotations.get("id", "") == target_id:
                return evt
        return None

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
        if isinstance(channel, int):
            channel = [channel]

        seg = self._load_segment(epochstreams, epoch_select)

        if channeltype in ("time", "timestamp", "t"):
            sig = seg.analogsignals[0]
            times = sig.times.magnitude
            return times[s0 - 1 : s1].reshape(-1, 1)

        data_list = []
        for ch in channel:
            sig = self._find_analog_by_ced_channel(seg, ch)
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
    ) -> tuple[Any, Any]:
        """Read events from CED files."""
        seg = self._load_segment(epochstreams, epoch_select)

        if isinstance(channel, int):
            channel = [channel]

        timestamps_all = []
        data_all = []

        for ch in channel:
            evt = self._find_event_by_ced_channel(seg, ch)
            if evt is not None:
                times = evt.times.magnitude
                mask = (times >= t0) & (times <= t1)
                timestamps_all.append(times[mask])
                if channeltype in ("marker", "mark", "mk"):
                    data_all.append(evt.labels[mask] if hasattr(evt, "labels") else np.array([]))
                else:
                    data_all.append(times[mask])
            else:
                timestamps_all.append(np.array([]))
                data_all.append(np.array([]))

        if len(channel) == 1:
            return timestamps_all[0], data_all[0]
        return timestamps_all, data_all

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List channels available for a given epoch."""
        seg = self._load_segment(epochstreams, epoch_select)

        channels: list[dict[str, Any]] = []

        for sig in seg.analogsignals:
            ch_ids = sig.array_annotations.get("channel_ids", [])
            if len(ch_ids) > 0:
                ced_ch = int(ch_ids[0]) + 1
            else:
                ced_ch = 0
            channels.append({"name": f"ai{ced_ch}", "type": "analog_in", "time_channel": 1})

        for evt in seg.events:
            ced_ch = int(evt.annotations.get("id", "0")) + 1
            name = evt.name if hasattr(evt, "name") else ""
            if "mark" in str(name).lower() or "keyboard" in str(name).lower():
                channels.append({"name": f"mk{ced_ch}", "type": "mark", "time_channel": 1})
            elif "text" in str(name).lower():
                channels.append({"name": f"text{ced_ch}", "type": "text", "time_channel": 1})
            else:
                channels.append({"name": f"e{ced_ch}", "type": "event", "time_channel": 1})

        return channels

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> float | np.ndarray:
        """Get the sample rate."""
        seg = self._load_segment(epochstreams, epoch_select)

        if isinstance(channel, int):
            channel = [channel]

        sr_list = []
        for ch in channel:
            try:
                sig = self._find_analog_by_ced_channel(seg, ch)
                sr_list.append(float(sig.sampling_rate.magnitude))
            except ValueError:
                sr_list.append(float("nan"))

        if len(sr_list) == 1:
            return sr_list[0]
        return np.array(sr_list)

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end times for an epoch."""
        seg = self._load_segment(epochstreams, epoch_select)

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
