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

        The way ``name`` is constructed depends on the reader's labeling
        convention for that channel type. See ``channelLabelingConvention``
        for the contract.

        Returns
        -------
        list of dict
            Each dict has keys: name, type, time_channel.
        """
        return []

    def channelLabelingConvention(self, channeltype: str) -> str:
        """Describe how this reader names channels of a given type.

        Returns a string declaring the naming convention this reader uses for
        channels of type ``channeltype`` in ``getchannelsepoch`` and as input
        to ``daqchannels2internalchannels``. One of:

        ``'indexed'``
            Names use NDR-standard prefixes (e.g. ``'ai'``, ``'ao'``, ``'ax'``,
            ``'di'``, ``'do'``, ``'t'``) followed by a 1-based count of
            recorded channels of that type. The first recorded analog input is
            ``'ai1'``, the second ``'ai2'``, and so on, regardless of any
            hardware-channel gaps in the underlying file. This is the
            convention NDI users typically expect; it is the default and the
            only one for which the trailing number is safe to interpret as a
            position.

        ``'physical'``
            Names use NDR-standard prefixes followed by the manufacturer's
            hardware channel number, in the manufacturer's own indexing base
            (which may be 0-based, 1-based, or per-type). The number is a
            hardware identity and may have gaps.

        ``'native'``
            The device-native string verbatim, which is opaque. The channel
            type must be taken from the ``'type'`` field of the
            ``getchannelsepoch`` entry, not parsed out of the name.

        Parameters
        ----------
        channeltype : str
            The channel type to describe.

        Returns
        -------
        str
            One of ``'indexed'``, ``'physical'``, or ``'native'``.
        """
        return "indexed"

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
    # Image / frame reading API
    # ------------------------------------------------------------------
    # The methods below define the frame-based reading interface used by
    # image-series readers (movies, z-stacks, slide scans). It is the imaging
    # counterpart of the regularly-sampled channel API above; the two families
    # are siblings, not subclasses. A reader that handles images implements
    # ONLY the frame API (not readchannels_epochsamples and friends); readers
    # that do not handle images inherit these no-op defaults.
    #
    # The frame API design is modeled on nansen.stack.ImageStack (VervaekeLab,
    # https://github.com/VervaekeLab/NANSEN). Only the design (method names,
    # dimension model) is adapted; no NANSEN source is used, so no NANSEN
    # dependency is introduced.
    #
    # Every method takes (epochstreams, epoch_select, ...) like the rest of the
    # reader API. A "frame" is one image plane along the ordering axes (T, and
    # Z when present). frameind indexes those ordering axes, 1-based, matching
    # the bridge's Semantic Parity policy for user-facing indices.

    def numframes(self, epochstreams: list[str], epoch_select: int = 1) -> int:
        """Return the number of frames in an image epoch.

        Modeled on ``nansen.stack.ImageStack`` NumTimepoints/NumPlanes.
        The abstract class returns 0.
        """
        return 0

    def framesize(self, epochstreams: list[str], epoch_select: int = 1) -> list[int]:
        """Return the ``[Y X C Z T]`` extent of an image epoch without reading pixels.

        Height, width, channels, z-planes, timepoints. Keeping the channel
        axis (C) separate from the spatial (Y, X) and ordering (Z, T) axes
        matches the V_delta axes+channels split.

        Modeled on ``nansen.stack.ImageStack/getFrameSetSize``. The abstract
        class returns zeros.
        """
        return [0, 0, 0, 0, 0]

    def dimensionorder(self, epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the dimension order of arrays returned by ``readframes``.

        A string over ``{Y, X, C, Z, T}``; the default is ``'YXCZT'``.

        Modeled on ``nansen.stack.ImageStack`` DimensionOrder/DataDimensionOrder.
        """
        return "YXCZT"

    def datatype(self, epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the underlying numeric class of the image pixels (e.g. ``'uint16'``).

        Modeled on ``nansen.stack.ImageStack`` DataType. The abstract class
        returns ``''``.
        """
        return ""

    def frametimes(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the time of each requested frame, in ``epochclock`` units.

        For a movie this is device-local time (seconds from the start of the
        epoch); for a clockless slide scan / z-stack the epoch clock is
        ``'no_time'`` and these are NaN.

        Modeled on ``nansen.stack.ImageStack/getFrameTimes``. The values
        returned here feed ``epochclock`` and ``t0_t1``. The abstract class
        returns an empty array.
        """
        return np.array([])

    def readframes(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        frameind: list[int] | np.ndarray | None = None,
        *,
        SelectC: list[int] | np.ndarray | None = None,
        SelectZ: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Read image frames from an epoch.

        Reads the timepoints indexed by ``frameind`` (1-based indices along
        the T axis) and returns them laid out in ``dimensionorder`` (default
        ``'YXCZT'``).

        ``SelectC`` and ``SelectZ`` select a subset of the channel (C) and
        plane (Z) axes, so the returned array is
        ``[Y, X, len(SelectC), len(SelectZ), len(frameind)]``. ``None`` keeps
        all of that axis. A reader may honor these by not reading the
        unselected data (e.g. skipping channel files); readers that cannot
        must post-select with ``selectframeCZ`` so the result is identical.

        Modeled on ``nansen.stack.ImageStack/getFrameSet``. The abstract class
        returns an empty array.
        """
        return np.array([])

    def metadata(self, epochstreams: list[str], epoch_select: int = 1) -> dict[str, Any]:
        """Return standardized image-acquisition metadata for an epoch.

        Describes HOW the frames were acquired — in particular the raster-scan
        timing that lets one compute when each line/pixel was sampled —
        separately from the pixel data itself. ALL TIME FIELDS ARE IN SECONDS.
        See ``emptyimagemetadata`` for the field set.

        A raster scan does not acquire a frame instantaneously: it sweeps line
        by line, so at slow frame rates the top of a frame is acquired well
        before the bottom. ``line_period`` (plus ``frametimes``) is what lets a
        caller reconstruct the acquisition time of each line/pixel.

        The abstract class returns the "empty" struct (``israster=False``, NaN
        timing). Raster readers override this and fill in the fields they can
        determine; fields that cannot be determined stay NaN.
        """
        return ndr_reader_base.emptyimagemetadata()

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def emptyimagemetadata() -> dict[str, Any]:
        """Return the standardized image-metadata dict with default (unknown) values.

        Every field at its "unknown" default: ``israster=False``,
        ``bidirectional=False``, and NaN for each timing/geometry value. A
        reader fills in the fields it can supply and leaves the rest at these
        defaults, so consumers always see the same field set. ALL TIME FIELDS
        ARE IN SECONDS.

        Fields
        ------
        israster : bool
            True if this epoch is a raster scan with known line/frame timing.
        frame_period : float
            Time to acquire one frame (s).
        line_period : float
            Time to acquire one scanned line/row (s).
        dwell_time : float
            Per-pixel dwell time (s).
        lines_per_frame : float
            Number of scanned lines (rows) per frame.
        pixels_per_line : float
            Number of pixels (columns) per line.
        bidirectional : bool
            True if alternate lines are scanned in the reverse direction.
        """
        return {
            "israster": False,
            "frame_period": float("nan"),
            "line_period": float("nan"),
            "dwell_time": float("nan"),
            "lines_per_frame": float("nan"),
            "pixels_per_line": float("nan"),
            "bidirectional": False,
        }

    @staticmethod
    def selectframeCZ(
        frames: np.ndarray,
        SelectC: list[int] | np.ndarray | None = None,
        SelectZ: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Post-select the channel (C) and plane (Z) axes of a frame array.

        Given a frame array in ``'YXCZT'`` order, returns the subset with
        channels ``SelectC`` (axis 2) and Z-planes ``SelectZ`` (axis 3).
        ``None`` keeps all of that axis. This is the shared helper readers use
        to honor ``readframes``' ``SelectC``/``SelectZ`` options when they
        cannot avoid reading the unselected data at the source.

        Selection indices are 1-based, matching ``readframes``.
        """
        if SelectC is not None and len(SelectC) > 0:
            idx = np.asarray(SelectC, dtype=int) - 1
            frames = frames[:, :, idx, :, :]
        if SelectZ is not None and len(SelectZ) > 0:
            idx = np.asarray(SelectZ, dtype=int) - 1
            frames = frames[:, :, :, idx, :]
        return frames

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
