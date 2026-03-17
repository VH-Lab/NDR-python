"""Neo-based data reading functions for NDR.

Port of +ndr/+format/+neo/neo_python.py (already Python in NDR-matlab).
"""

from __future__ import annotations

import numpy as np

from ndr.format.neo import Utils


def getchannelsepoch(filenames, segment_index, block_index=1):
    """Get channels for an epoch from the given files."""

    def format_channels(channels):
        return [
            {
                "name": ch["name"],
                "type": Utils.channel_type_from_neo_to_ndr(ch["_type"]),
            }
            for ch in channels
        ]

    if segment_index == "all":
        raw_reader = Utils.get_raw_reader(filenames)
        all_channels = Utils.get_header_channels(raw_reader)
        return format_channels(all_channels)
    else:
        reader = Utils.get_reader(filenames)
        raw_reader = Utils.get_raw_reader(filenames)
        segment_channels = Utils.get_channels_from_segment(
            reader, raw_reader, int(segment_index) - 1, int(block_index) - 1
        )
        return format_channels(segment_channels)


def readchannels_epochsamples(
    channel_type, channel_names, filenames, segment_index, start_sample, end_sample, block_index=1
):
    """Read channel data for a range of samples."""
    if channel_type == "time":
        raw_reader = Utils.get_raw_reader(filenames)
        sample_rate = Utils.get_sample_rates(raw_reader, channel_names)[0]
        sample_interval = 1 / sample_rate

        times = []
        for sample in range(int(start_sample) - 1, int(end_sample)):
            times.append([sample * sample_interval])
        return np.array(times, np.float32)
    else:
        raw_reader = Utils.get_raw_reader(filenames)
        stream_index = Utils.from_channel_names_to_stream_index(raw_reader, channel_names)

        raw = raw_reader.get_analogsignal_chunk(
            block_index=int(block_index) - 1,
            seg_index=int(segment_index) - 1,
            i_start=int(start_sample) - 1,
            i_stop=int(end_sample),
            channel_names=channel_names,
            stream_index=int(stream_index),
        )
        rescaled = raw_reader.rescale_signal_raw_to_float(
            raw, channel_names=channel_names, stream_index=int(stream_index)
        )
        return rescaled


def daqchannels2internalchannels(channel_names, filenames, segment_index, block_index=1):
    """Convert DAQ channel names to internal channel descriptions."""
    reader = Utils.get_reader(filenames)
    raw_reader = Utils.get_raw_reader(filenames)

    channels = Utils.get_channels_from_segment(
        reader, raw_reader, int(segment_index) - 1, int(block_index) - 1
    )
    needed_channels = [ch for ch in channels if ch["name"] in channel_names]

    return [
        {
            "internal_type": ch["_type"],
            "internal_number": ch["id"],
            "internal_channelname": ch["name"],
            "ndr_type": Utils.channel_type_from_neo_to_ndr(ch["_type"]),
            "samplerate": str(Utils.channel_to_sample_rate(ch)),
            "stream_id": ch["stream_id"],
        }
        for ch in needed_channels
    ]


def canbereadtogether(channelstruct):
    """Check if all channels belong to the same stream."""
    stream_ids = [ch["stream_id"] for ch in channelstruct]
    unique_stream_ids = list(set(stream_ids))

    if len(unique_stream_ids) > 1:
        error_message = ""
        for ch in channelstruct:
            error_message += f"\nChannel_name: '{ch['name']}', stream_id: '{ch['stream_id']}'."
        return {
            "b": 0,
            "errormsg": f"All of your channels should belong to a single signal_stream in Neo.\n{error_message}\n",
        }
    else:
        return {"b": 1, "errormsg": ""}


def samplerate(filenames, channel_names):
    """Get sample rates for the given channel names."""
    raw_reader = Utils.get_raw_reader(filenames)
    return Utils.get_sample_rates(raw_reader, channel_names)


def t0_t1(filenames, segment_index, block_index=1):
    """Get the start and stop times for a segment."""
    reader = Utils.get_reader(filenames)
    block = reader.read()[block_index - 1]
    segment = block.segments[segment_index - 1]

    def get_magnitude(q):
        return q.rescale("s").item()

    return [get_magnitude(segment.t_start), get_magnitude(segment.t_stop)]


def readevents_epochsamples_native(
    channel_type, channel_names, filenames, segment_index, start_time, end_time, block_index=1
):
    """Read event or marker data for a time range."""
    raw_reader = Utils.get_raw_reader(filenames)

    if channel_type == "marker":
        list_of_timestamps = []
        list_of_marker_codes = []

        for channel_name in channel_names:
            event_channel_index = Utils.from_channel_name_to_event_channel_index(
                raw_reader, channel_name
            )
            timestamps, _durations, marker_codes = raw_reader.get_event_timestamps(
                event_channel_index=event_channel_index,
                block_index=int(block_index) - 1,
                seg_index=int(segment_index) - 1,
                t_start=start_time,
                t_stop=end_time,
            )
            timestamps = raw_reader.rescale_event_timestamp(
                timestamps, event_channel_index=event_channel_index
            )
            list_of_timestamps.append(timestamps.tolist())
            list_of_marker_codes.append(marker_codes.tolist())

        return [list_of_timestamps, list_of_marker_codes]
    elif channel_type == "event":
        list_of_timestamps = []
        list_of_events = []

        for channel_name in channel_names:
            timestamps = raw_reader.get_spike_timestamps(
                spike_channel_index=Utils.from_channel_name_to_marker_channel_index(
                    raw_reader, channel_name
                ),
                block_index=int(block_index) - 1,
                seg_index=int(segment_index) - 1,
                t_start=start_time,
                t_stop=end_time,
            )
            timestamps = raw_reader.rescale_spike_timestamp(timestamps)
            events = [1] * len(timestamps)
            list_of_timestamps.append(timestamps.tolist())
            list_of_events.append(events)

        return [list_of_timestamps, list_of_events]
    else:
        raise ValueError(f"channel_type should be either 'marker' or 'event', not {channel_type}")
