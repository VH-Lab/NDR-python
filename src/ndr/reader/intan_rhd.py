"""Intan RHD reader class.

Port of +ndr/+reader/intan_rhd.m
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.intan.read_Intan_RHD2000_datafile import (
    Intan_RHD2000_blockinfo,
    read_Intan_RHD2000_datafile,
)
from ndr.format.intan.read_Intan_RHD2000_header import read_Intan_RHD2000_header
from ndr.reader.base import ndr_reader_base


class ndr_reader_intan__rhd(ndr_reader_base):
    """Reader for Intan Technologies .RHD file format.

    Port of ndr.reader.intan_rhd.
    """

    def __init__(self) -> None:
        super().__init__()

    def read(
        self,
        epochstreams: list[str],
        channelstring: str,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read data using channel string, supporting Intan native names.

        Accepts standard NDR channel strings (e.g., 'ai1-3') or Intan
        native channel names (e.g., 'A021', 'A000+A001').
        """
        from ndr.string.channelstring2channels import channelstring2channels

        if options is None:
            options = {}
        epoch_select = options.get("epoch_select", 1)
        use_samples = options.get("useSamples", 0)
        s0 = options.get("s0", None)
        s1 = options.get("s1", None)

        channelprefix, channelnumber = channelstring2channels(channelstring)
        channelstruct = self.daqchannels2internalchannels(
            channelprefix, channelnumber, epochstreams, epoch_select
        )

        if not channelstruct:
            raise ValueError(f"Could not resolve channels from '{channelstring}'.")

        internal_type = channelstruct[0]["internal_type"]
        channels = [ch["internal_number"] for ch in channelstruct]

        if not use_samples or s0 is None or s1 is None:
            t0t1 = self.t0_t1(epochstreams, epoch_select)
            sr = channelstruct[0]["samplerate"]
            s0 = round(1 + t0t1[0][0] * sr)
            s1 = round(1 + t0t1[0][1] * sr)

        data = self.readchannels_epochsamples(
            internal_type, channels, epochstreams, epoch_select, int(s0), int(s1)
        )
        time = self.readchannels_epochsamples(
            "time", channels, epochstreams, epoch_select, int(s0), int(s1)
        )
        return data, time

    def daqchannels2internalchannels(
        self,
        channelprefix: list[str],
        channelnumber: list[int] | np.ndarray,
        epochstreams: list[str],
        epoch_select: int = 1,
    ) -> list[dict[str, Any]]:
        """Convert DAQ channel prefixes/numbers to internal channel structures."""
        channelstruct: list[dict[str, Any]] = []

        filename = self.filenamefromepochfiles(epochstreams)[0]
        header = read_Intan_RHD2000_header(filename)

        for c in range(len(channelnumber)):
            intan_type, absolute = ndr_reader_intan__rhd.intananychannelname2intanchanneltype(
                channelprefix[c]
            )
            ndr_type = ndr_reader_intan__rhd.intanchanneltype2mfdaqchanneltype(intan_type)
            header_name = ndr_reader_intan__rhd.mfdaqchanneltype2intanheadertype(ndr_type)
            header_chunk = header.get(header_name, [])

            entry: dict[str, Any] = {
                "internal_type": ndr_type,
                "internal_number": 0,
                "internal_channelname": "",
                "ndr_type": ndr_type,
                "samplerate": 0.0,
            }

            if not absolute:
                entry["internal_number"] = channelnumber[c]
                native_names = [ch["native_channel_name"] for ch in header_chunk]
                if channelnumber[c] - 1 < len(native_names):
                    entry["internal_channelname"] = native_names[channelnumber[c] - 1]
                entry["samplerate"] = self.samplerate(
                    epochstreams, epoch_select, channelprefix[c], channelnumber[c]
                )
            else:
                chan_name = f"{channelprefix[c]}-{channelnumber[c]:03d}"
                entry["internal_channelname"] = chan_name
                native_names = [ch["native_channel_name"] for ch in header_chunk]
                if chan_name in native_names:
                    entry["internal_number"] = native_names.index(chan_name) + 1
                else:
                    raise ValueError(
                        f"Requested channel {chan_name} was not recorded in this file."
                    )
                entry["samplerate"] = self.samplerate(
                    epochstreams,
                    epoch_select,
                    entry["ndr_type"],
                    entry["internal_number"],
                )

            channelstruct.append(entry)

        return channelstruct

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end epoch times."""
        filename, parentdir, isdirectory = self.filenamefromepochfiles(epochstreams)
        header = read_Intan_RHD2000_header(filename)

        if not isdirectory:
            _blockinfo, _bpb, _bp, num_data_blocks = Intan_RHD2000_blockinfo(filename, header)
            total_samples = 60 * num_data_blocks
        else:
            time_dat = Path(parentdir) / "time.dat"
            if not time_dat.exists():
                raise FileNotFoundError(
                    f"File time.dat necessary in directory {parentdir} but not found."
                )
            total_samples = time_dat.stat().st_size // 4

        sr = header["frequency_parameters"]["amplifier_sample_rate"]
        total_time = total_samples / sr
        t0 = 0.0
        t1 = total_time - 1.0 / sr

        return [[t0, t1]]

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List channels available for a given epoch."""
        intan_channel_types = [
            "amplifier_channels",
            "aux_input_channels",
            "board_dig_in_channels",
            "board_dig_out_channels",
        ]

        filename = self.filenamefromepochfiles(epochstreams)[0]
        header = read_Intan_RHD2000_header(filename)

        channels: list[dict[str, Any]] = []

        # Time channel 1 (amplifier rate)
        channels.append({"name": "t1", "type": "time", "time_channel": 1})

        # Time channel 2 for aux (if aux channels present)
        if header.get("aux_input_channels"):
            channels.append({"name": "t2", "type": "time", "time_channel": 2})

        for intan_type in intan_channel_types:
            if intan_type in header and header[intan_type]:
                channel_type_entry = ndr_reader_intan__rhd.intanheadertype2mfdaqchanneltype(
                    intan_type
                )
                for ch in header[intan_type]:
                    name = ndr_reader_intan__rhd.intanname2mfdaqname(channel_type_entry, ch)
                    time_channel = 2 if channel_type_entry == "auxiliary_in" else 1
                    channels.append(
                        {
                            "name": name,
                            "type": channel_type_entry,
                            "time_channel": time_channel,
                        }
                    )

        return channels

    def underlying_datatype(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str,
        channel: int | list[int],
    ) -> tuple[str, np.ndarray, int]:
        """Get the underlying data type for a channel."""
        if channeltype in ("analog_in", "analog_out"):
            return "uint16", np.array([32768, 0.195]), 16
        elif channeltype == "auxiliary_in":
            return "uint16", np.array([0, 3.74e-05]), 16
        elif channeltype == "time":
            return "float64", np.array([0, 1]), 64
        elif channeltype in ("digital_in", "digital_out"):
            return "char", np.array([0, 1]), 8
        elif channeltype in ("eventmarktext", "event", "marker", "text"):
            return "float64", np.array([0, 1]), 64
        else:
            raise ValueError(f"Unknown channel type '{channeltype}'.")

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
        filename, parentdir, isdirectory = self.filenamefromepochfiles(epochstreams)

        intanchanneltype = ndr_reader_intan__rhd.mfdaqchanneltype2intanchanneltype(channeltype)

        if isinstance(channel, int):
            channel = [channel]

        sr = self.samplerate(epochstreams, epoch_select, channeltype, channel[0])
        sr_arr = np.atleast_1d(np.asarray(sr, dtype=float))
        sr_unique = np.unique(sr_arr)
        if len(sr_unique) != 1:
            raise ValueError("Do not know how to handle different sampling rates across channels.")
        sr_val = float(sr_unique[0])

        t0 = (s0 - 1) / sr_val
        t1 = (s1 - 1) / sr_val

        if intanchanneltype == "time":
            channel = [1]

        if not isdirectory:
            data = read_Intan_RHD2000_datafile(filename, "", intanchanneltype, channel, t0, t1)
        else:
            # Directory-based reading (one-file-per-channel mode)
            # For now, delegate to the single file reader
            raise NotImplementedError("Directory-based Intan reading not yet implemented.")

        return data

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read events (not supported for Intan RHD, returns empty)."""
        return np.array([]), np.array([])

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int,
        channeltype: str | list[str],
        channel: int | list[int],
    ) -> float | np.ndarray:
        """Get the sample rate for specific channels."""
        if epoch_select != 1:
            raise ValueError("Intan RHD files have 1 epoch per file.")

        filename = self.filenamefromepochfiles(epochstreams)[0]
        header = read_Intan_RHD2000_header(filename)

        if isinstance(channel, int):
            channel = [channel]

        sr_list = []
        for i in range(len(channel)):
            if isinstance(channeltype, list):
                ct = channeltype[i]
            else:
                ct = channeltype

            freq_field = ndr_reader_intan__rhd.mfdaqchanneltype2intanfreqheader(ct)
            sr_list.append(header["frequency_parameters"][freq_field])

        if len(sr_list) == 1:
            return sr_list[0]
        return np.array(sr_list)

    def filenamefromepochfiles(self, filename_array: list[str]) -> tuple[str, str, bool]:
        """Find the .rhd file from a list of epoch files.

        Returns
        -------
        tuple of (filename, parentdir, isdirectory)
        """
        rhd_files = [f for f in filename_array if f.lower().endswith(".rhd")]

        if len(rhd_files) > 1:
            raise ValueError("Need only 1 .rhd file per epoch.")
        elif len(rhd_files) == 0:
            raise ValueError("Need 1 .rhd file per epoch.")

        filename = rhd_files[0]
        parentdir = str(Path(filename).parent)
        isdirectory = False

        # Check if this is the one-file-per-channel format
        if Path(filename).stem == "info":
            time_dat_files = [f for f in filename_array if f.endswith("time.dat")]
            if time_dat_files:
                isdirectory = True

        return filename, parentdir, isdirectory

    # ------------------------------------------------------------------
    # Static helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def mfdaqchanneltype2intanheadertype(channeltype: str) -> str:
        """Convert ndr.reader.base channel types to Intan header field names."""
        mapping = {
            "analog_in": "amplifier_channels",
            "ai": "amplifier_channels",
            "digital_in": "board_dig_in_channels",
            "di": "board_dig_in_channels",
            "digital_out": "board_dig_out_channels",
            "do": "board_dig_out_channels",
            "auxiliary": "aux_input_channels",
            "aux": "aux_input_channels",
            "ax": "aux_input_channels",
            "auxiliary_in": "aux_input_channels",
            "auxiliary_input": "aux_input_channels",
        }
        if channeltype not in mapping:
            raise ValueError(f"Could not convert channeltype '{channeltype}'.")
        return mapping[channeltype]

    @staticmethod
    def intanheadertype2mfdaqchanneltype(intanchanneltype: str) -> str:
        """Convert Intan header field names to ndr channel types."""
        mapping = {
            "amplifier_channels": "analog_in",
            "board_dig_in_channels": "digital_in",
            "board_dig_out_channels": "digital_out",
            "aux_input_channels": "auxiliary_in",
        }
        if intanchanneltype not in mapping:
            raise ValueError(f"Could not convert channeltype '{intanchanneltype}'.")
        return mapping[intanchanneltype]

    @staticmethod
    def mfdaqchanneltype2intanchanneltype(channeltype: str) -> str:
        """Convert ndr channel type to Intan internal channel type."""
        mapping = {
            "analog_in": "amp",
            "ai": "amp",
            "digital_in": "din",
            "di": "din",
            "digital_out": "dout",
            "do": "dout",
            "time": "time",
            "timestamp": "time",
            "auxiliary": "aux",
            "aux": "aux",
            "auxiliary_input": "aux",
            "auxiliary_in": "aux",
        }
        ct = channeltype.lower()
        if ct not in mapping:
            raise ValueError(f"Do not know how to convert channel type '{channeltype}'.")
        return mapping[ct]

    @staticmethod
    def intanchanneltype2mfdaqchanneltype(channeltype: str) -> str:
        """Convert Intan internal channel type to ndr channel type."""
        mapping = {
            "amp": "ai",
            "din": "di",
            "dout": "do",
            "time": "time",
            "aux": "ai",
        }
        ct = channeltype.lower()
        if ct not in mapping:
            raise ValueError(f"Do not know how to convert channel type '{channeltype}'.")
        return mapping[ct]

    @staticmethod
    def intanname2mfdaqname(channel_type: str, name_or_struct: str | dict[str, Any]) -> str:
        """Convert Intan native channel name to ndr.reader.base format."""
        if isinstance(name_or_struct, dict):
            name = name_or_struct["native_channel_name"]
            chip_channel = name_or_struct.get("chip_channel")
        else:
            name = name_or_struct
            chip_channel = None

        chan = None

        # Try to find separator '-'
        sep_idx = name.rfind("-")
        if sep_idx >= 0:
            try:
                chan_intan = int(name[sep_idx + 1 :])
                chan = chan_intan + 1  # Intan numbers from 0
            except ValueError:
                pass
        else:
            # Try to find trailing digits
            m = re.search(r"\d+$", name)
            if m:
                chan_intan = int(m.group())
                if channel_type.startswith("aux") or channel_type.startswith("ax"):
                    chan = chan_intan  # Aux channels are 1-based
                else:
                    chan = chan_intan + 1

        if chan is None and chip_channel is not None:
            chan = chip_channel + 1

        prefix = ndr_reader_base.mfdaq_prefix(channel_type)
        if chan is None:
            return prefix
        return f"{prefix}{chan}"

    @staticmethod
    def mfdaqchanneltype2intanfreqheader(channeltype: str) -> str:
        """Return the header field name for frequency info for a channel type."""
        mapping = {
            "analog_in": "amplifier_sample_rate",
            "ai": "amplifier_sample_rate",
            "digital_in": "board_dig_in_sample_rate",
            "di": "board_dig_in_sample_rate",
            "time": "amplifier_sample_rate",
            "timestamp": "amplifier_sample_rate",
            "auxiliary": "aux_input_sample_rate",
            "aux": "aux_input_sample_rate",
            "auxiliary_in": "aux_input_sample_rate",
        }
        ct = channeltype.lower()
        if ct not in mapping:
            raise ValueError(f"Do not know frequency header for channel type '{channeltype}'.")
        return mapping[ct]

    @staticmethod
    def intananychannelname2intanchanneltype(
        intananychannelname: str,
    ) -> tuple[str, bool]:
        """Convert any channel name to Intan channel type.

        Returns
        -------
        tuple of (str, bool)
            (intan_channel_type, is_absolute_reference)
        """
        # First try as a standard ndr channel type
        try:
            return (
                ndr_reader_intan__rhd.mfdaqchanneltype2intanchanneltype(intananychannelname),
                False,
            )
        except ValueError:
            pass

        # Try as Intan bank name
        name_lower = intananychannelname.lower()
        if name_lower in ("a", "b", "c", "d"):
            return "amp", True
        elif name_lower in ("aaux", "baux", "caux", "daux"):
            return "aux", True
        elif name_lower in ("avdd1", "bvdd1", "cvdd1", "dvdd1"):
            return "supply", True
        else:
            raise ValueError(f"Do not know how to convert channel bank '{intananychannelname}'.")
