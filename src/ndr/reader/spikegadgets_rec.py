"""SpikeGadgets REC reader class.

Port of +ndr/+reader/spikegadgets_rec.m

Two places deliberately depart from the MATLAB source, both documented at the
method that departs and in the bridge YAML:

* ``t0_t1`` computes the epoch length from the true packet layout. MATLAB's
  version predates the stride correction and is wrong by ~17 ms on the bundled
  fixture; mirroring it would contradict the format layer this reader sits on.
* the auxiliary and digital paths of ``readchannels_epochsamples`` work here.
  MATLAB's read the ``startbyte``/``bit``/``number`` fields off the result of
  ``getchannelsepoch``, which strips them, so those branches raise before
  reading anything.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ndr.format.spikegadgets.read_rec_analogChannels import read_rec_analogChannels
from ndr.format.spikegadgets.read_rec_config import read_rec_config
from ndr.format.spikegadgets.read_rec_configsize import read_rec_configsize
from ndr.format.spikegadgets.read_rec_digitalChannels import read_rec_digitalChannels
from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels
from ndr.reader.base import ndr_reader_base
from ndr.time.clocktype import ClockType

# Channels per nTrode. MATLAB hardcodes 4 with a note that it could be
# generalised; kept the same so the channel lists agree.
CHANNELS_PER_NTRODE = 4

# MCU digital inputs are numbered above the 32 non-MCU inputs.
MCU_DIGITAL_OFFSET = 32


class ndr_reader_spikegadgets__rec(ndr_reader_base):
    """Reader for the SpikeGadgets .rec file format.

    Port of ndr.reader.spikegadgets_rec.
    """

    def __init__(self) -> None:
        super().__init__()

    def channelLabelingConvention(self, channeltype: str) -> str:
        """Return the channel naming convention for this reader.

        Names use the integer parsed out of the native ``'Ain%d'``/``'Din%d'``
        identifiers, which is the hardware channel index rather than a 1-based
        position into the recorded set.

        See ``ndr.reader.base.ndr_reader_base.channelLabelingConvention``.
        """
        return "physical"

    def filenamefromepochfiles(self, epochstreams: list[str], epoch_select: int = 1) -> str:
        """Return the single ``.rec`` file for the epoch."""
        pattern = re.compile(r".*\.rec$", re.IGNORECASE)
        matches = [f for f in epochstreams if pattern.match(str(f))]

        if not matches:
            raise ValueError("Need 1 .rec file per epoch.")
        if len(matches) > 1:
            raise ValueError("Need only 1 .rec file per epoch.")
        return str(matches[0])

    # -- channel table -----------------------------------------------------

    @staticmethod
    def _describe_header_channel(entry: dict[str, Any]) -> dict[str, Any]:
        """Map one native header channel onto its NDR name, type and number.

        Mirrors the name parsing in getchannelsepoch.m: 'Ain%d' and 'Aout%d'
        are auxiliary, 'Din%d' and 'Dout%d' are digital, and anything else is
        an MCU digital input whose number is offset past the non-MCU inputs.
        """
        native = entry["name"]

        if native.startswith("Ai"):
            number = int(re.match(r"Ain(\d+)", native).group(1))
            return {"name": f"axn{number}", "type": "auxiliary", "number": number}
        if native.startswith("Ao"):
            number = int(re.match(r"Aout(\d+)", native).group(1))
            return {"name": f"axo{number}", "type": "auxiliary", "number": number}
        if native.startswith("Di"):
            number = int(re.match(r"Din(\d+)", native).group(1))
            return {"name": f"di{number}", "type": "digital_in", "number": number}
        if native.startswith("Do"):
            number = int(re.match(r"Dout(\d+)", native).group(1))
            return {"name": f"do{number}", "type": "digital_out", "number": number}

        number = int(re.match(r"MCU_Din(\d+)", native).group(1)) + MCU_DIGITAL_OFFSET
        return {"name": f"di{number}", "type": "digital_in", "number": number}

    def _internal_channels(self, epochstreams: list[str], epoch_select: int = 1) -> list[dict]:
        """The full channel table, including the byte/bit addressing.

        getchannelsepoch strips startbyte, bit and number before returning, so
        the read paths need this. MATLAB calls getchannelsepoch and then reads
        the stripped fields off the result, which is why its auxiliary and
        digital branches cannot work.
        """
        filename = self.filenamefromepochfiles(epochstreams, epoch_select)
        fileconfig, header_channels = read_rec_config(filename)

        channels: list[dict[str, Any]] = []
        for entry in header_channels:
            described = self._describe_header_channel(entry)
            described["startbyte"] = entry.get("startbyte")
            described["bit"] = entry.get("bit")
            channels.append(described)

        for ntrode in fileconfig.get("nTrodes", []):
            for info in ntrode["channelInfo"][:CHANNELS_PER_NTRODE]:
                number = int(info["packetLocation"]) + 1
                channels.append(
                    {
                        "name": f"ai{number}",
                        "type": "analog_in",
                        "number": number,
                        "startbyte": None,
                        "bit": None,
                    }
                )

        channels.sort(key=lambda c: (c["type"], c["number"]))
        return channels

    def getchannelsepoch(
        self, epochstreams: list[str], epoch_select: int = 1
    ) -> list[dict[str, Any]]:
        """List the channels available in this epoch.

        Returns name/type/time_channel entries sorted by type then number,
        with a leading 't1' time channel, matching getchannelsepoch.m.
        """
        internal = self._internal_channels(epochstreams, epoch_select)

        channels = [{"name": "t1", "type": "time", "time_channel": 1}]
        channels += [{"name": c["name"], "type": c["type"], "time_channel": 1} for c in internal]
        return channels

    # -- timing ------------------------------------------------------------

    def samplerate(
        self,
        epochstreams: list[str],
        epoch_select: int = 1,
        channeltype: str | None = None,
        channel: int | list[int] | None = None,
    ) -> float:
        """Return the sample rate, which is the same for every channel here."""
        filename = self.filenamefromepochfiles(epochstreams, epoch_select)
        fileconfig, _channels = read_rec_config(filename)
        return float(fileconfig["samplingRate"])

    def t0_t1(self, epochstreams: list[str], epoch_select: int = 1) -> list[list[float]]:
        """Return the beginning and end times of the epoch, in seconds.

        Diverges from t0_t1 in spikegadgets_rec.m, which sizes the packet from
        headerSize + 2 + channels and subtracts only headerSize from the file
        length. Both are wrong: the packet is headerSize + 4 + channels (the
        + 2 is fread's skip argument, not a width), and what precedes the
        packet stream is the whole configuration block, not the header. On the
        bundled example.rec that yields 60090.7 packets -- a fractional count --
        and an epoch 17 ms too long. Mirroring it would contradict the format
        layer this reader calls.
        """
        filename = self.filenamefromepochfiles(epochstreams, epoch_select)
        fileconfig, _channels = read_rec_config(filename)

        header_bytes = int(fileconfig["headerSize"]) * 2
        channel_bytes = int(fileconfig["numChannels"]) * 2
        packet_bytes = header_bytes + 4 + channel_bytes

        data_bytes = Path(filename).stat().st_size - read_rec_configsize(filename)
        total_samples = data_bytes // packet_bytes

        sr = float(fileconfig["samplingRate"])
        return [[0.0, (total_samples - 1) / sr]]

    def epochclock(self, epochstreams: list[str], epoch_select: int = 1) -> list[ClockType]:
        """Return the clock types available for this epoch."""
        return [ClockType("dev_local_time")]

    # -- reading -----------------------------------------------------------

    def _byteandbit(
        self, epochstreams: list[str], epoch_select: int, channeltype: str, channel: list[int]
    ) -> list[list[int]]:
        """Resolve channel numbers to the (startbyte, bit) pairs the readers want.

        Returned in the caller's requested order rather than the table's, so a
        non-ascending request is not silently permuted.
        """
        internal = self._internal_channels(epochstreams, epoch_select)
        by_number = {c["number"]: c for c in internal if c["type"] == channeltype}

        pairs = []
        for number in channel:
            entry = by_number.get(int(number))
            if entry is None:
                raise ValueError(
                    f"Channel {number} of type '{channeltype}' is not recorded in this epoch."
                )
            bit = 0 if entry["bit"] is None else int(entry["bit"]) + 1
            pairs.append([int(entry["startbyte"]), bit])
        return pairs

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int = 1,
        s0: int = 1,
        s1: int = 1,
    ) -> np.ndarray:
        """Read samples from the given channels.

        ``channel`` is interpreted per channeltype, as in the MATLAB version:
        analog_in/analog_out are packet positions, auxiliary and digital are
        the numbers parsed out of the native names.
        """
        if isinstance(channel, (int, np.integer)):
            channel = [int(channel)]
        channel = [int(c) for c in np.atleast_1d(channel)]

        # Normalise through the base prefix map so the short forms ('ai', 'ax',
        # 'di', 't') work alongside the long ones. MATLAB compares the long
        # strings directly and rejects 'ai'; accepting both is the convention
        # elsewhere in this port and is what the generic reader test uses.
        prefix = self.mfdaq_prefix(channeltype)

        filename = self.filenamefromepochfiles(epochstreams, epoch_select)
        fileconfig, _channels = read_rec_config(filename)
        num_channels = int(fileconfig["numChannels"])
        header_size = int(fileconfig["headerSize"])
        sr = float(fileconfig["samplingRate"])

        if prefix in ("ai", "ao"):
            data, _ts = read_rec_trodeChannels(
                filename, num_channels, [c - 1 for c in channel], sr, header_size, s0, s1
            )
            return np.asarray(data)

        if prefix == "t":
            _data, ts = read_rec_trodeChannels(
                filename, num_channels, [c - 1 for c in channel], sr, header_size, s0, s1
            )
            return np.asarray(ts).reshape(-1, 1)

        if prefix == "ax":
            pairs = self._byteandbit(epochstreams, epoch_select, "auxiliary", channel)
            data, _ts = read_rec_analogChannels(
                filename,
                num_channels,
                [p[0] for p in pairs],
                sr,
                header_size,
                s0,
                s1,
                True,
            )
            return np.asarray(data).T

        if prefix in ("di", "do"):
            long_type = "digital_in" if prefix == "di" else "digital_out"
            pairs = self._byteandbit(epochstreams, epoch_select, long_type, channel)
            data, _ts = read_rec_digitalChannels(
                filename, num_channels, np.array(pairs), sr, header_size, s0, s1, True
            )
            return np.asarray(data).T

        raise ValueError(f"Unknown channel type '{channeltype}' for the SpikeGadgets reader.")

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SpikeGadgets .rec files carry no event channels."""
        return np.array([]), np.array([])
