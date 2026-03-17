"""Abstract base class for all NDR readers.

Port of +ndr/+reader/base.m
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ndr.time.clocktype import ClockType
from ndr.time.fun.samples2times import samples2times as _samples2times
from ndr.time.fun.times2samples import times2samples as _times2samples


class ndr_reader_base(ABC):
    """Abstract base class for Neuroscience Data Readers.

    All format-specific readers inherit from this class and must implement
    the abstract methods.

    Port of ndr.reader.base. MATLAB: ndr.reader.base
    """

    def __init__(self) -> None:
        self.MightHaveTimeGaps: bool = False

    # ------------------------------------------------------------------
    # Concrete methods (base provides default implementations)
    # ------------------------------------------------------------------

    def canbereadtogether(self, channelstruct: list[dict[str, Any]]) -> tuple[bool, str]:
        """Check if channels in a channel struct can be read in a single call.

        Parameters
        ----------
        channelstruct : list of dict
            Each dict has keys: internal_type, internal_number,
            internal_channelname, ndr_type, samplerate.

        Returns
        -------
        tuple of (bool, str)
            (True, '') if channels can be read together, or
            (False, error_message) otherwise.
        """
        b = True
        errormsg = ""

        sr = [ch["samplerate"] for ch in channelstruct]
        sr_arr = np.array(sr, dtype=float)

        if not np.all(np.isnan(sr_arr)):
            # If all are not NaN, then none can be
            if np.any(np.isnan(sr_arr)):
                b = False
                errormsg = (
                    "All samplerates must either be the same number or they must "
                    "all be NaN, indicating they are all not regularly sampled channels."
                )
            else:
                sr_unique = np.unique(sr_arr)
                if len(sr_unique) != 1:
                    b = False
                    errormsg = (
                        "All sample rates must be the same for all requested "
                        "regularly-sampled channels for a single function call."
                    )

        return b, errormsg

    def daqchannels2internalchannels(
        self,
        channelprefix: list[str],
        channelnumber: list[int] | np.ndarray,
        epochstreams: list[str],
        epoch_select: int = 1,
    ) -> list[dict[str, Any]]:
        """Convert DAQ channel prefixes and numbers to internal channel structures.

        Parameters
        ----------
        channelprefix : list of str
            Channel prefixes describing channels for this device.
        channelnumber : array-like of int
            Channel numbers, one per entry in channelprefix.
        epochstreams : list of str
            File paths comprising the epoch of data.
        epoch_select : int
            Which epoch in the file to access (usually 1).

        Returns
        -------
        list of dict
            Each dict has keys: internal_type, internal_number,
            internal_channelname, ndr_type, samplerate.
        """
        # Abstract class returns empty
        return []

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return the clock types available for this epoch.

        Parameters
        ----------
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.

        Returns
        -------
        list of ClockType
            Clock types for this epoch.
        """
        return [ClockType("dev_local_time")]

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List channels available for a given epoch.

        Parameters
        ----------
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.

        Returns
        -------
        list of dict
            Each dict has keys: name, type, time_channel.
        """
        return []

    def underlying_datatype(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> tuple[str, np.ndarray, int]:
        """Get the underlying data type for a channel in an epoch.

        Parameters
        ----------
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        channeltype : str
            The type of channel.
        channel : int or list of int
            Channel number(s).

        Returns
        -------
        tuple of (str, numpy.ndarray, int)
            (datatype, polynomial_coefficients, datasize_in_bits)
        """
        if isinstance(channel, int):
            n_channels = 1
        else:
            n_channels = len(channel)

        if channeltype in ("analog_in", "analog_out", "auxiliary_in", "time"):
            datatype = "float64"
            datasize = 64
            p = np.tile([0, 1], (n_channels, 1))
        elif channeltype in ("digital_in", "digital_out"):
            datatype = "char"
            datasize = 8
            p = np.tile([0, 1], (n_channels, 1))
        elif channeltype in ("eventmarktext", "event", "marker", "text"):
            datatype = "float64"
            datasize = 64
            p = np.tile([0, 1], (n_channels, 1))
        else:
            raise ValueError(f"Unknown channel type '{channeltype}'.")

        return datatype, p, datasize

    @abstractmethod
    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Read data from specified channels.

        Parameters
        ----------
        channeltype : str
            Type of channel to read.
        channel : int or list of int
            Channel number(s) to read (1-based).
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        s0 : int
            Start sample number (1-based).
        s1 : int
            End sample number (1-based).

        Returns
        -------
        numpy.ndarray
            Data array with one column per channel.
        """
        ...

    @abstractmethod
    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read events or markers for specified channels.

        Parameters
        ----------
        channeltype : str
            Type of channel to read.
        channel : int or list of int
            Channel number(s) to read.
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        t0 : float
            Start time.
        t1 : float
            End time.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            (timestamps, data)
        """
        ...

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> np.ndarray | float:
        """Get the sample rate for specific channels.

        Parameters
        ----------
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        channeltype : str
            Type of channel.
        channel : int or list of int
            Channel number(s).

        Returns
        -------
        numpy.ndarray or float
            Sample rate(s) in Hz.
        """
        return np.array([])

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end epoch times.

        Parameters
        ----------
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.

        Returns
        -------
        list of list
            [[t0, t1]] for each clock type. Abstract class returns [[NaN, NaN]].
        """
        return [[float("nan"), float("nan")]]

    def read(
        self,
        epochstreams: list[str],
        channelstring: str,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read data and time from specified channels.

        Convenience method that parses a channel string and calls the
        appropriate low-level read method.

        Parameters
        ----------
        epochstreams : list of str
            File paths comprising the epoch.
        channelstring : str
            Channel specification (e.g., 'ai1-3', 'A021', 'e22').
        options : dict, optional
            Options dict with keys: epoch_select (int), useSamples (bool),
            s0 (int), s1 (int), t0 (float), t1 (float).

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            (data, time)
        """
        from ndr.string.channelstring2channels import channelstring2channels

        if options is None:
            options = {}
        epoch_select = options.get("epoch_select", 1)
        use_samples = options.get("useSamples", 0)
        s0 = options.get("s0", None)
        s1 = options.get("s1", None)
        t0 = options.get("t0", None)
        t1 = options.get("t1", None)

        channelprefix, channelnumber = channelstring2channels(channelstring)

        if not channelprefix:
            raise ValueError(f"Could not parse channel string '{channelstring}'.")

        ndr_type = self.mfdaq_type(channelprefix[0])

        if ndr_type in ("analog_in", "analog_out", "time", "ax"):
            if use_samples and s0 is not None and s1 is not None:
                pass
            else:
                t0t1 = self.t0_t1(epochstreams, epoch_select)
                sr = self.samplerate(epochstreams, epoch_select, channelprefix[0], channelnumber[0])
                actual_t0 = t0 if t0 is not None else t0t1[0][0]
                actual_t1 = t1 if t1 is not None else t0t1[0][1]
                s0 = round(1 + actual_t0 * sr)
                s1 = round(1 + actual_t1 * sr)

            data = self.readchannels_epochsamples(
                channelprefix[0], channelnumber, epochstreams, epoch_select, int(s0), int(s1)
            )
            time = self.readchannels_epochsamples(
                "time", channelnumber, epochstreams, epoch_select, int(s0), int(s1)
            )
            return data, time
        else:
            t0t1 = self.t0_t1(epochstreams, epoch_select)
            actual_t0 = t0 if t0 is not None else t0t1[0][0]
            actual_t1 = t1 if t1 is not None else t0t1[0][1]
            return self.readevents_epochsamples_native(
                ndr_type, channelnumber, epochstreams, epoch_select, actual_t0, actual_t1
            )

    def samples2times(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s: np.ndarray | int | float,
    ) -> np.ndarray:
        """Convert sample numbers to time.

        Parameters
        ----------
        channeltype : str
            Type of channel.
        channel : int or list of int
            Channel number(s).
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        s : array-like
            Sample numbers (1-based).

        Returns
        -------
        numpy.ndarray
            Times in seconds.
        """
        sr = self.samplerate(epochstreams, epoch_select, channeltype, channel)
        sr_arr = np.atleast_1d(np.asarray(sr, dtype=float))
        sr_unique = np.unique(sr_arr)
        if len(sr_unique) != 1:
            raise ValueError("Do not know how to handle different sampling rates across channels.")
        t0t1 = self.t0_t1(epochstreams, epoch_select)
        return _samples2times(s, t0t1[0], sr_unique[0])

    def times2samples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t: np.ndarray | float,
    ) -> np.ndarray:
        """Convert time to sample numbers.

        Parameters
        ----------
        channeltype : str
            Type of channel.
        channel : int or list of int
            Channel number(s).
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        t : array-like
            Times in seconds.

        Returns
        -------
        numpy.ndarray
            Sample numbers (1-based).
        """
        sr = self.samplerate(epochstreams, epoch_select, channeltype, channel)
        sr_arr = np.atleast_1d(np.asarray(sr, dtype=float))
        sr_unique = np.unique(sr_arr)
        if len(sr_unique) != 1:
            raise ValueError("Do not know how to handle different sampling rates across channels.")
        t0t1 = self.t0_t1(epochstreams, epoch_select)
        return _times2samples(t, t0t1[0], sr_unique[0])

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def mfdaq_channeltypes() -> list[str]:
        """Return supported channel types for multifunction DAQ readers.

        Returns
        -------
        list of str
            Channel type strings.
        """
        return [
            "analog_in",
            "aux_in",
            "analog_out",
            "digital_in",
            "digital_out",
            "marker",
            "event",
            "time",
        ]

    @staticmethod
    def mfdaq_prefix(channeltype: str) -> str:
        """Return the channel prefix for a given channel type.

        Parameters
        ----------
        channeltype : str
            The channel type string.

        Returns
        -------
        str
            The channel prefix (e.g., 'ai', 'di', 't').
        """
        prefix_map = {
            "analog_in": "ai",
            "ai": "ai",
            "analog_out": "ao",
            "ao": "ao",
            "digital_in": "di",
            "di": "di",
            "digital_out": "do",
            "do": "do",
            "digital_in_event": "dep",
            "digital_in_event_pos": "dep",
            "de": "dep",
            "dep": "dep",
            "digital_in_event_neg": "den",
            "den": "den",
            "digital_in_mark": "dimp",
            "digital_in_mark_pos": "dimp",
            "dim": "dimp",
            "dimp": "dimp",
            "digital_in_mark_neg": "dimn",
            "dimn": "dimn",
            "time": "t",
            "timestamp": "t",
            "t": "t",
            "auxiliary": "ax",
            "aux": "ax",
            "ax": "ax",
            "auxiliary_in": "ax",
            "marker": "mk",
            "mark": "mk",
            "mk": "mk",
            "event": "e",
            "e": "e",
            "metadata": "md",
            "md": "md",
            "text": "text",
        }
        if channeltype not in prefix_map:
            raise ValueError(f"Unknown channel type '{channeltype}'.")
        return prefix_map[channeltype]

    @staticmethod
    def mfdaq_type(channeltype: str) -> str:
        """Return the preferred long channel type name for a given channel type.

        Parameters
        ----------
        channeltype : str
            The channel type string (short or long form).

        Returns
        -------
        str
            The canonical long channel type name.
        """
        type_map = {
            "analog_in": "analog_in",
            "ai": "analog_in",
            "analog_out": "analog_out",
            "ao": "analog_out",
            "digital_in": "digital_in",
            "di": "digital_in",
            "digital_out": "digital_out",
            "do": "digital_out",
            "time": "time",
            "timestamp": "time",
            "t": "time",
            "auxiliary": "ax",
            "aux": "ax",
            "ax": "ax",
            "auxiliary_in": "ax",
            "marker": "mark",
            "mark": "mark",
            "mk": "mark",
            "event": "event",
            "e": "event",
            "text": "text",
        }
        if channeltype not in type_map:
            raise ValueError(f"Type '{channeltype}' is unknown.")
        return type_map[channeltype]
