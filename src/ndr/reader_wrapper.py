"""High-level NDR reader wrapper.

Port of +ndr/reader.m

This module provides the Reader class that wraps format-specific reader
subclasses and provides a unified interface for reading neural data.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ndr.fun.ndrresource import ndrresource
from ndr.reader.base import Base
from ndr.string.channelstring2channels import channelstring2channels
from ndr.time.clocktype import ClockType


class Reader:
    """High-level Neuroscience Data Reader.

    Wraps a format-specific reader (subclass of ndr.reader.base.Base)
    and provides a unified interface.

    Port of ndr.reader (the MATLAB class, not the package).

    Parameters
    ----------
    ndr_reader_type : str
        Data format identifier (e.g., 'intan', 'rhd', 'abf', 'smr').
    """

    def __init__(self, ndr_reader_type: str) -> None:
        j = ndrresource("ndr_reader_types.json")
        match = None
        for entry in j:
            if ndr_reader_type.lower() in [t.lower() for t in entry["type"]]:
                match = entry
                break

        if match is None:
            raise ValueError(f"Do not know how to make a reader of type '{ndr_reader_type}'.")

        # Dynamically import and instantiate the reader class
        classname = match["classname"]
        parts = classname.rsplit(".", 1)
        module_path = parts[0]
        class_name = parts[1]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        self.ndr_reader_base: Base = cls()

    def read(
        self,
        epochstreams: list[str],
        channelstring: str | list[str],
        *,
        t0: float = float("-inf"),
        t1: float = float("inf"),
        epoch_select: int = 1,
        useSamples: bool = False,
        s0: float = float("nan"),
        s1: float = float("nan"),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read data or time information from specified channels and epoch.

        Parameters
        ----------
        epochstreams : list of str
            File paths comprising the epoch.
        channelstring : str or list of str
            Channel specification (e.g., 'ai1-3,5', 'ai1+di1').
        t0 : float
            Start time in seconds (-inf for earliest).
        t1 : float
            End time in seconds (inf for latest).
        epoch_select : int
            Epoch index within the file (usually 1).
        useSamples : bool
            If True, interpret s0/s1 as sample numbers instead of times.
        s0 : float
            Start sample number (1-based) if useSamples is True.
        s1 : float
            End sample number (1-based) if useSamples is True.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            (data, time)
        """
        is_neo = type(self.ndr_reader_base).__name__ == "NeoReader"

        if is_neo:
            channelstruct = self.ndr_reader_base.daqchannels2internalchannels(
                [], channelstring, epochstreams, epoch_select
            )
        else:
            channelprefix, channelnumber = channelstring2channels(channelstring)
            channelstruct = self.ndr_reader_base.daqchannels2internalchannels(
                channelprefix, channelnumber, epochstreams, epoch_select
            )

        b, errormsg = self.ndr_reader_base.canbereadtogether(channelstruct)

        if not b:
            raise ValueError(
                "Specified channels cannot be read in a single function call. "
                f"Please split channel reading by similar channel types. {errormsg}"
            )

        ndr_type = channelstruct[0]["ndr_type"]

        if ndr_type in (
            "analog_input",
            "analog_output",
            "analog_in",
            "analog_out",
            "ai",
            "ao",
        ):
            if not useSamples:
                sr = channelstruct[0]["samplerate"]
                s0 = round(1 + t0 * sr)
                s1 = round(1 + t1 * sr)

            if is_neo:
                channels = channelstring
            else:
                channels = [ch["internal_number"] for ch in channelstruct]

            data = self.readchannels_epochsamples(
                channelstruct[0]["internal_type"],
                channels,
                epochstreams,
                epoch_select,
                int(s0),
                int(s1),
            )
            time = self.readchannels_epochsamples(
                "time",
                channels,
                epochstreams,
                epoch_select,
                int(s0),
                int(s1),
            )
        else:
            if is_neo:
                channels = channelstring
            else:
                channels = [ch["internal_number"] for ch in channelstruct]

            data, time = self.readevents_epochsamples(
                [ch["internal_type"] for ch in channelstruct],
                channels,
                epochstreams,
                epoch_select,
                t0,
                t1,
            )

        return data, time

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return the clock types for an epoch."""
        return self.ndr_reader_base.epochclock(epochstreams, epoch_select)

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end times for an epoch."""
        return self.ndr_reader_base.t0_t1(epochstreams, epoch_select)

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List channels available for a given epoch."""
        return self.ndr_reader_base.getchannelsepoch(epochstreams, epoch_select)

    def underlying_datatype(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> tuple[str, np.ndarray, int]:
        """Get the native data type for specified channels."""
        return self.ndr_reader_base.underlying_datatype(
            epochstreams, epoch_select, channeltype, channel
        )

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Read regularly sampled data channels."""
        return self.ndr_reader_base.readchannels_epochsamples(
            channeltype, channel, epochstreams, epoch_select, s0, s1
        )

    def readevents_epochsamples(
        self,
        channeltype: list[str],
        channel: list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[Any, Any]:
        """Read event/marker data or derive events from digital channels."""
        derived_types = {"dep", "den", "dimp", "dimn"}

        if set(channeltype) & derived_types:
            timestamps_list: list[np.ndarray] = []
            data_list: list[np.ndarray] = []

            for i, ch in enumerate(channel):
                srd = self.samplerate(epochstreams, epoch_select, "di", ch)
                s0d = 1 + round(srd * t0)
                s1d = 1 + round(srd * t1)

                data_here = self.readchannels_epochsamples(
                    "di", [ch], epochstreams, epoch_select, s0d, s1d
                )
                time_here = self.readchannels_epochsamples(
                    "time", [ch], epochstreams, epoch_select, s0d, s1d
                )

                data_flat = data_here.flatten()
                time_flat = time_here.flatten()

                ct = channeltype[i]
                if ct in ("dep", "dimp"):
                    on_samples = np.where((data_flat[:-1] == 0) & (data_flat[1:] == 1))[0]
                    if ct == "dimp":
                        off_samples = np.where((data_flat[:-1] == 1) & (data_flat[1:] == 0))[0] + 1
                    else:
                        off_samples = np.array([], dtype=int)
                elif ct in ("den", "dimn"):
                    on_samples = np.where((data_flat[:-1] == 1) & (data_flat[1:] == 0))[0]
                    if ct == "dimn":
                        off_samples = np.where((data_flat[:-1] == 0) & (data_flat[1:] == 1))[0] + 1
                    else:
                        off_samples = np.array([], dtype=int)
                else:
                    on_samples = np.array([], dtype=int)
                    off_samples = np.array([], dtype=int)

                ts = np.concatenate([time_flat[on_samples], time_flat[off_samples]])
                d = np.concatenate(
                    [
                        np.ones(len(on_samples)),
                        -np.ones(len(off_samples)),
                    ]
                )

                if len(off_samples) > 0:
                    order = np.argsort(ts)
                    ts = ts[order]
                    d = d[order]

                timestamps_list.append(ts)
                data_list.append(d)

            if len(channel) == 1:
                return timestamps_list[0], data_list[0]
            return timestamps_list, data_list
        else:
            return self.readevents_epochsamples_native(
                channeltype, channel, epochstreams, epoch_select, t0, t1
            )

    def readevents_epochsamples_native(
        self,
        channeltype: list[str],
        channel: list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[Any, Any]:
        """Read native event/marker channels."""
        return self.ndr_reader_base.readevents_epochsamples_native(
            channeltype, channel, epochstreams, epoch_select, t0, t1
        )

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> Any:
        """Get the sample rate for specific channels."""
        return self.ndr_reader_base.samplerate(epochstreams, epoch_select, channeltype, channel)

    def samples2times(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s: np.ndarray | int | float,
    ) -> np.ndarray:
        """Convert sample numbers to time."""
        return self.ndr_reader_base.samples2times(
            channeltype, channel, epochstreams, epoch_select, s
        )

    def times2samples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t: np.ndarray | float,
    ) -> np.ndarray:
        """Convert time to sample numbers."""
        return self.ndr_reader_base.times2samples(
            channeltype, channel, epochstreams, epoch_select, t
        )

    @property
    def MightHaveTimeGaps(self) -> bool:
        """Whether the underlying reader might have time gaps."""
        return self.ndr_reader_base.MightHaveTimeGaps
