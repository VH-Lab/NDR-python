"""Neuropixels SpikeGLX reader class.

Port of +ndr/+reader/neuropixelsGLX.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.binarymatrix.read import read as binarymatrix_read
from ndr.format.neuropixelsGLX.header import header
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType
from ndr.time.fun.samples2times import samples2times


class ndr_reader_neuropixelsGLX(ndr_reader_base):
    """Reader for Neuropixels SpikeGLX AP-band data.

    This class reads action-potential band data from Neuropixels probes
    acquired with the SpikeGLX software. Each instance handles one probe's
    AP stream (one .ap.bin / .ap.meta file pair per epoch).

    SpikeGLX saves Neuropixels data as flat interleaved int16 binary files
    with companion .meta text files. The binary files have no header.
    Channel count, sample rate, and gain information are read from the
    .meta file.

    Channel mapping:
        - Neural channels are exposed as 'analog_in' (ai1..aiN)
        - The sync word is exposed as 'digital_in' (di1)
        - A single time channel 't1' is always present

    Data is returned as int16 to preserve native precision. Use
    :func:`ndr.format.neuropixelsGLX.samples2volts` for voltage conversion.

    Port of ndr.reader.neuropixelsGLX.
    """

    def __init__(self) -> None:
        super().__init__()

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return the clock type objects for an epoch.

        Returns ``[ClockType('dev_local_time')]`` since SpikeGLX timestamps
        are relative to the start of each file.
        """
        return [ClockType("dev_local_time")]

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end epoch times.

        Duration is computed from the binary file size, channel count,
        and sample rate.
        """
        metafile = self.filenamefromepochfiles(epochstreams)
        info = header(metafile)

        binfile = Path(metafile[:-4] + "bin")
        if binfile.is_file():
            bytes_per_sample = 2 * info["n_saved_chans"]
            total_samples = binfile.stat().st_size // bytes_per_sample
        else:
            total_samples = round(info["file_time_secs"] * info["sample_rate"])

        t_end = (total_samples - 1) / info["sample_rate"]
        return [[0.0, t_end]]

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List channels available for a given epoch.

        Neural channels are 'analog_in' (ai1..aiN), the sync channel is
        'digital_in' (di1), and a time channel 't1' is always present.
        """
        metafile = self.filenamefromepochfiles(epochstreams)
        info = header(metafile)

        channels: list[dict[str, Any]] = []

        # Time channel
        channels.append({"name": "t1", "type": "time", "time_channel": 1})

        # Neural channels (analog_in)
        for i in range(1, info["n_neural_chans"] + 1):
            channels.append({"name": f"ai{i}", "type": "analog_in", "time_channel": 1})

        # Sync channel (digital_in)
        if info["n_sync_chans"] > 0:
            channels.append({"name": "di1", "type": "digital_in", "time_channel": 1})

        return channels

    def underlying_datatype(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> tuple[str, np.ndarray, int]:
        """Get the native data type for channels.

        For analog_in: int16, [0 1], 16 bits.
        For time: double (computed), [0 1], 64 bits.
        For digital_in: int16 (sync word), [0 1], 16 bits.
        """
        if isinstance(channel, int):
            n_channels = 1
        else:
            n_channels = len(channel)

        ct = channeltype.lower()
        if ct in ("analog_in", "ai"):
            datatype = "int16"
            datasize = 16
            p = np.tile([0, 1], (n_channels, 1))
        elif ct in ("time", "t"):
            datatype = "float64"
            datasize = 64
            p = np.tile([0, 1], (n_channels, 1))
        elif ct in ("digital_in", "di"):
            datatype = "int16"
            datasize = 16
            p = np.tile([0, 1], (n_channels, 1))
        else:
            return super().underlying_datatype(epochstreams, epoch_select, channeltype, channel)

        return datatype, p, datasize

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Read data samples for specified channels.

        Reads data between sample s0 and s1 (inclusive, 1-based).

        For 'analog_in': returns int16 neural data.
        For 'time': returns float64 time stamps in seconds.
        For 'digital_in': returns int16 sync word values.
        """
        metafile = self.filenamefromepochfiles(epochstreams)
        info = header(metafile)
        binfile = metafile[:-4] + "bin"

        ct = channeltype.lower()

        if ct in ("time", "timestamp", "t"):
            t0t1 = self.t0_t1(epochstreams, epoch_select)
            data = samples2times(np.arange(s0, s1 + 1, dtype=float), t0t1[0], info["sample_rate"])
            return data.reshape(-1, 1) if data.ndim == 1 else data

        elif ct in ("analog_in", "ai"):
            # channel numbers are 1-based and map directly to file columns
            if isinstance(channel, int):
                channel = [channel]
            channel_arr = np.array(channel, dtype=np.int64)
            data, _, _, _ = binarymatrix_read(
                binfile,
                info["n_saved_chans"],
                channel_arr,
                float(s0),
                float(s1),
                dataType="int16",
                byteOrder="ieee-le",
                headerSkip=0,
            )
            return data

        elif ct in ("digital_in", "di"):
            # Sync channel is the last channel in the file
            sync_chan = np.array([info["n_saved_chans"]], dtype=np.int64)
            data, _, _, _ = binarymatrix_read(
                binfile,
                info["n_saved_chans"],
                sync_chan,
                float(s0),
                float(s1),
                dataType="int16",
                byteOrder="ieee-le",
                headerSkip=0,
            )
            return data

        else:
            raise ValueError(f'Unknown channel type "{channeltype}".')

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read events or markers. Neuropixels has no native event channels.

        Returns empty arrays since Neuropixels data is purely
        regularly-sampled.
        """
        return np.array([]), np.array([])

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str | list[str],
        channel: int | list[int],
    ) -> np.ndarray | float:
        """Get the sample rate for specified channels.

        For Neuropixels AP-band, all channels share the same sample rate
        (typically 30 kHz).
        """
        metafile = self.filenamefromepochfiles(epochstreams)
        info = header(metafile)
        if isinstance(channel, (list, np.ndarray)):
            return np.full(len(channel), info["sample_rate"])
        return info["sample_rate"]

    def filenamefromepochfiles(self, filename_array: list[str]) -> str:
        """Identify the .ap.meta file from epoch file list.

        Searches the list for a file matching *.ap.meta. Errors if zero
        or more than one match is found.
        """
        matches = [f for f in filename_array if f.lower().endswith(".ap.meta")]

        if len(matches) == 0:
            raise FileNotFoundError("No .ap.meta file found in the epoch file list.")
        if len(matches) > 1:
            raise ValueError("Multiple .ap.meta files found. Each epoch should have exactly one.")
        return matches[0]

    def daqchannels2internalchannels(
        self,
        channelprefix: list[str],
        channelnumber: list[int] | np.ndarray,
        epochstreams: list[str],
        epoch_select: int = 1,
    ) -> list[dict[str, Any]]:
        """Convert DAQ channel specs to internal format."""
        from ndr.string.channelstring2channels import channelstring2channels

        channelstruct: list[dict[str, Any]] = []
        channels_available = self.getchannelsepoch(epochstreams, epoch_select)

        for i in range(len(channelnumber)):
            current_prefix = channelprefix[i].lower()
            current_number = channelnumber[i]

            for ch_avail in channels_available:
                avail_prefix, avail_number = channelstring2channels(ch_avail["name"])
                if (
                    avail_prefix
                    and avail_prefix[0].lower() == current_prefix
                    and avail_number[0] == current_number
                ):
                    sr = self.samplerate(epochstreams, epoch_select, current_prefix, current_number)
                    channelstruct.append(
                        {
                            "internal_channelname": ch_avail["name"],
                            "internal_type": ch_avail["type"],
                            "internal_number": current_number,
                            "ndr_type": self.mfdaq_type(ch_avail["type"]),
                            "samplerate": (
                                float(sr) if not isinstance(sr, np.ndarray) else float(sr.item())
                            ),
                        }
                    )
                    break

        return channelstruct
