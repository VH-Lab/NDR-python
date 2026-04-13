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
    """Reader for SpikeGLX data (AP, LF, and NIDQ streams).

    This class reads data from Neuropixels probes and NI-DAQ devices
    acquired with the SpikeGLX software. Each instance handles one
    stream (one .bin / .meta file pair per epoch).

    SpikeGLX saves data as flat interleaved int16 binary files with
    companion .meta text files. The binary files have no header.
    Channel count, sample rate, and gain information are read from
    the .meta file.

    Channel mapping:
        - Analog channels are exposed as 'analog_in' (ai1..aiN).
          For AP streams these are neural probe channels; for NIDQ
          streams these are NI-DAQ analog inputs (MN + MA + XA).
        - Digital lines are exposed as 'digital_in' (di1..diM),
          where each di channel is a single bit of the packed digital
          word(s). For NIDQ streams the count comes from
          ``8 * (niXDBytes1 + niXDBytes2)``; for IMEC streams it is
          ``16 * n_sync_chans``.
        - A single time channel 't1' is always present.

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

        Analog channels are 'analog_in' (ai1..aiN), digital lines are
        'digital_in' (di1..diM) with one entry per single-bit line in
        the packed digital word(s), and a time channel 't1' is always
        present.
        """
        metafile = self.filenamefromepochfiles(epochstreams)
        info = header(metafile)

        channels: list[dict[str, Any]] = []

        # Time channel
        channels.append({"name": "t1", "type": "time", "time_channel": 1})

        # Analog channels (analog_in)
        for i in range(1, info["n_neural_chans"] + 1):
            channels.append({"name": f"ai{i}", "type": "analog_in", "time_channel": 1})

        # Digital lines (digital_in) — one per bit of the packed digital word(s)
        for i in range(1, info["n_digital_lines"] + 1):
            channels.append({"name": f"di{i}", "type": "digital_in", "time_channel": 1})

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

        For 'analog_in': returns int16 data (neural for AP, NI-DAQ analog
        inputs for NIDQ).
        For 'time': returns float64 time stamps in seconds.
        For 'digital_in': returns int16 single-bit values (0 or 1)
        extracted from the packed digital word(s). ``channel`` gives
        the 1-based digital line(s).
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
            if isinstance(channel, int):
                channel = [channel]
            line_idx = np.array(channel, dtype=int)

            if np.any(line_idx < 1) or np.any(line_idx > info["n_digital_lines"]):
                raise ValueError(
                    f"Digital line out of range; valid lines are 1..{info['n_digital_lines']}."
                )

            # Digital word columns are at the end of each sample row
            first_dw_col = info["n_saved_chans"] - info["n_digital_word_cols"] + 1

            # Look up the (column, bit) position for each requested line
            # line_idx is 1-based; digital_line_col/bit arrays are 0-indexed
            col_offsets = info["digital_line_col"][line_idx - 1]
            bit_pos = info["digital_line_bit"][line_idx - 1]

            n_samples = int(s1) - int(s0) + 1
            data = np.zeros((n_samples, len(channel)), dtype=np.int16)

            unique_cols = np.unique(col_offsets)
            for uc in unique_cols:
                file_col = np.array([first_dw_col + uc], dtype=np.int64)
                raw, _, _, _ = binarymatrix_read(
                    binfile,
                    info["n_saved_chans"],
                    file_col,
                    float(s0),
                    float(s1),
                    dataType="int16",
                    byteOrder="ieee-le",
                    headerSkip=0,
                )
                # raw is (n_samples, 1) int16 — treat as uint16 for bit extraction
                raw_uint = raw[:, 0].view(np.uint16)
                mask = col_offsets == uc
                indices = np.where(mask)[0]
                for idx in indices:
                    data[:, idx] = ((raw_uint >> bit_pos[idx]) & 1).astype(np.int16)

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
        """Read events or markers. SpikeGLX has no native event channels.

        Returns empty arrays since SpikeGLX data is purely
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

        All channels in a single SpikeGLX binary file share one sample
        rate: typically 30 kHz for AP, 2.5 kHz for LF, or ~25 kHz for NIDQ.
        """
        metafile = self.filenamefromepochfiles(epochstreams)
        info = header(metafile)
        if isinstance(channel, (list, np.ndarray)):
            return np.full(len(channel), info["sample_rate"])
        return info["sample_rate"]

    def filenamefromepochfiles(self, filename_array: list[str]) -> str:
        """Identify the companion .meta file from the epoch file list.

        First searches for a ``.bin`` file and derives the ``.meta`` path
        from it (replacing the last 3 characters). This allows multiple
        ``.meta`` files (e.g. both AP and NIDQ) to coexist in the same
        epoch file list — the ``.bin`` file disambiguates which stream
        this reader instance handles.

        If no ``.bin`` file is present, falls back to finding a single
        ``.meta`` file. Errors if zero or more than one ``.meta`` match
        is found without a ``.bin`` to disambiguate.
        """
        # Primary path: find the .bin file and derive .meta from it
        binfile = ""
        for f in filename_array:
            if f.lower().endswith(".bin"):
                binfile = f
                break

        if binfile:
            metafile = binfile[:-3] + "meta"
            # Verify the .meta exists in the file list or on disk
            if metafile not in filename_array and not Path(metafile).is_file():
                raise FileNotFoundError(f"No companion .meta file found for {binfile}.")
            return metafile

        # Fallback: find a single .meta file
        meta_matches = [f for f in filename_array if f.lower().endswith(".meta")]
        if len(meta_matches) == 0:
            raise FileNotFoundError("No .meta file found in the epoch file list.")
        if len(meta_matches) > 1:
            raise ValueError(
                "Multiple .meta files found and no .bin file to disambiguate."
            )
        return meta_matches[0]

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
