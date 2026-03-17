"""Read configuration from a SpikeGadgets .rec file.

Port of +ndr/+format/+spikegadgets/read_rec_config.m
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def read_rec_config(filename: str | Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Read the configuration from a SpikeGadgets ``.rec`` file.

    Parameters
    ----------
    filename : str or Path
        Path to the ``.rec`` file.

    Returns
    -------
    out : dict
        Configuration dict with fields including ``configText``,
        ``numChannels``, ``samplingRate``, ``headerSize``, ``nTrodes``,
        ``headerChannels``, etc.
    officialchannels : list of dict
        List of all hardware channels with fields ``name``, ``type``,
        ``startbyte``, ``bit``.
    """
    filename = Path(filename)

    with open(filename, "rb") as f:
        junk = f.read(1000000)

    config_end_tag = b"</Configuration>"
    idx = junk.find(config_end_tag)
    if idx < 0:
        raise ValueError("Could not find </Configuration> tag in file.")
    config_size = idx + len(config_end_tag)

    config_text = junk[:config_size]

    # Parse XML
    root = ET.fromstring(config_text.decode("utf-8", errors="replace"))

    out: dict[str, Any] = {}
    out["configText"] = config_text

    # Global options
    global_opts = root.find("GlobalOptions") or root.find("GlobalConfiguration")
    if global_opts is not None:
        for k, v in global_opts.attrib.items():
            out[k] = v

    # Hardware configuration
    hw = root.find("HardwareConfiguration")
    device_list: list[dict[str, Any]] = []
    officialchannels: list[dict[str, str]] = []

    total_header_size = 1  # sync byte

    if hw is not None:
        for k, v in hw.attrib.items():
            out[k] = v

        for device_elem in hw.findall("Device"):
            dev_info: dict[str, Any] = dict(device_elem.attrib)
            dev_channels: list[dict[str, str]] = []

            if int(dev_info.get("available", "0")) == 1:
                total_header_size += int(dev_info.get("numBytes", "0"))

                for ch_elem in device_elem.findall("Channel"):
                    ch_info = dict(ch_elem.attrib)
                    dev_channels.append(ch_info)
                    officialchannels.append(
                        {
                            "name": ch_info.get("id", ""),
                            "type": ch_info.get("dataType", ""),
                            "startbyte": ch_info.get("startByte", ""),
                            "bit": ch_info.get("bit", ""),
                        }
                    )

            dev_info["channels"] = dev_channels
            device_list.append(dev_info)

    out["headerSize"] = str(total_header_size // 2)

    # Spike configuration / nTrodes
    spike_config = root.find("SpikeConfiguration")
    num_cards = (
        int(out.get("numChannels", "0")) // 32 if int(out.get("numChannels", "0")) > 0 else 1
    )
    complete_channel_list = []

    if spike_config is not None:
        out["nTrodes"] = []
        for ntrode_elem in spike_config.findall("SpikeNTrode"):
            ntrode: dict[str, Any] = dict(ntrode_elem.attrib)
            ntrode["channelInfo"] = []

            for spike_ch in ntrode_elem.findall("SpikeChannel"):
                ch_info = dict(spike_ch.attrib)
                hw_chan = int(ch_info.get("hwChan", "0"))
                new_hw_chan = (hw_chan % 32) * num_cards + hw_chan // 32
                complete_channel_list.append(new_hw_chan)
                ch_info["packetLocation"] = new_hw_chan
                ntrode["channelInfo"].append(ch_info)

            out["nTrodes"].append(ntrode)

    # Adjust for unsaved channels
    num_chan_in = int(out.get("numChannels", "0"))
    complete_channel_list.sort()
    unused = set(range(num_chan_in)) - set(complete_channel_list)
    num_chan_in_file = num_chan_in - len(unused)
    out["numChannels"] = str(num_chan_in_file)

    if out.get("saveDisplayedChanOnly") == "1" and unused:
        for ntrode in out.get("nTrodes", []):
            for ch_info in ntrode.get("channelInfo", []):
                hw_chan = ch_info["packetLocation"]
                idx_in_list = complete_channel_list.index(hw_chan)
                skipped = sum(
                    complete_channel_list[i] - complete_channel_list[i - 1] - 1
                    for i in range(1, idx_in_list + 1)
                    if complete_channel_list[i] - complete_channel_list[i - 1] > 1
                )
                ch_info["packetLocation"] = hw_chan - skipped

    # Auxiliary/header channels
    aux_config = root.find("AuxDisplayConfiguration") or root.find("HeaderDisplay")
    if aux_config is not None:
        out["headerChannels"] = []
        for ch_elem in aux_config:
            if ch_elem.tag in ("HeaderChannel", "DispChannel"):
                ch_info = dict(ch_elem.attrib)

                # Look up startByte from device list
                dev_name = ch_info.get("device", "")
                ch_id = ch_info.get("id", "")
                if dev_name and ch_id:
                    offset_so_far = 1
                    sorted_devs = sorted(
                        [d for d in device_list if int(d.get("available", "0")) == 1],
                        key=lambda d: int(d.get("packetOrderPreference", "0")),
                    )
                    for dev in sorted_devs:
                        if dev.get("name") == dev_name:
                            for hw_ch in dev.get("channels", []):
                                if hw_ch.get("id") == ch_id:
                                    sb = int(hw_ch.get("startByte", "0")) + offset_so_far
                                    ch_info["startByte"] = str(sb)
                                    ch_info.update(hw_ch)
                                    break
                            break
                        offset_so_far += int(dev.get("numBytes", "0"))

                out["headerChannels"].append(ch_info)

    return out, officialchannels
