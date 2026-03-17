"""Neo library channel utilities.

Port of +ndr/+format/+neo/Utils.py (already Python in NDR-matlab).
"""

from __future__ import annotations

try:
    import neo
    from neo.rawio.cedrawio import CedRawIO
except ImportError:  # pragma: no cover
    neo = None  # type: ignore[assignment]
    CedRawIO = None  # type: ignore[assignment]


def get_header_channels(raw_reader):
    """Get all channels from a Neo raw reader's parsed header."""
    raw_reader.parse_header()
    header = raw_reader.header
    all_channels = []
    for _type in ["signal_channels", "spike_channels", "event_channels"]:
        numpy_channels = header[_type]
        python_channels = [dict(zip(numpy_channels.dtype.names, x)) for x in numpy_channels]
        for python_channel in python_channels:
            python_channel["_type"] = _type
            all_channels.append(python_channel)
    return all_channels


def get_channels_from_segment(reader, raw_reader, segment_index, block_index):
    """Get channels present in a specific segment."""
    blocks = reader.read(lazy=True)
    block = blocks[block_index]
    segment = block.segments[segment_index]

    signals = segment.analogsignals + segment.spiketrains + segment.irregularlysampledsignals
    channel_names = []
    for signal in signals:
        channel_names += signal.array_annotations["channel_names"].tolist()

    header_channels = get_header_channels(raw_reader)
    our_channels = [ch for ch in header_channels if ch["name"] in channel_names]
    return our_channels


def from_channel_names_to_stream_index(raw_reader, channel_names):
    """Return the stream_index that the first passed channel belongs to."""
    all_channels = get_header_channels(raw_reader)
    channel = [ch for ch in all_channels if ch["name"] == channel_names[0]][0]
    stream_id = channel["stream_id"]

    all_streams = raw_reader.header["signal_streams"]
    for index, stream in enumerate(all_streams):
        if stream_id == stream["id"]:
            return index
    return None


def from_channel_name_to_event_channel_index(raw_reader, channel_name):
    """Return the event channel index for a given channel name."""
    raw_reader.parse_header()
    event_channels = raw_reader.header["event_channels"]
    n = raw_reader.event_channels_count()
    for event_channel_index in range(n):
        if event_channels[event_channel_index]["name"] == channel_name:
            return event_channel_index
    return None


def from_channel_name_to_marker_channel_index(raw_reader, channel_name):
    """Return the spike/marker channel index for a given channel name."""
    raw_reader.parse_header()
    spike_channels = raw_reader.header["spike_channels"]
    n = raw_reader.spike_channels_count()
    for spike_channel_index in range(n):
        if spike_channels[spike_channel_index]["name"] == channel_name:
            return spike_channel_index
    return None


def channel_to_sample_rate(channel):
    """Get the sample rate for a channel dict."""
    if channel["_type"] == "signal_channels":
        return channel["sampling_rate"]
    elif channel["_type"] == "spike_channels":
        return channel["wf_sampling_rate"]
    elif channel["_type"] == "event_channels":
        return 0
    return 0


def channel_type_from_neo_to_ndr(_type):
    """Convert a Neo channel type to an NDR channel type string."""
    if _type == "signal_channels":
        return "analog_input"
    elif _type == "spike_channels":
        return "event"
    elif _type == "event_channels":
        return "marker"
    return "unknown"


def get_sample_rates(raw_reader, channel_names):
    """Get sample rates for the given channel names."""
    header_channels = get_header_channels(raw_reader)
    our_channels = [ch for ch in header_channels if ch["name"] in channel_names]
    return [channel_to_sample_rate(ch) for ch in our_channels]


def get_reader(filenames):
    """Get a Neo IO reader for the given filenames."""
    reader = neo.io.get_io(filenames[0])
    return reader


def get_raw_reader(filenames):
    """Get a Neo RawIO reader for the given filenames."""
    filename = filenames[0]

    if filename.endswith(".smr"):
        return CedRawIO(filename=filename)

    Klass = neo.rawio.get_rawio_class(filename)
    if Klass.rawmode in ("one-file", "multi-file"):
        raw_reader = Klass(filename=filename)
    elif Klass.rawmode == "one-dir":
        raw_reader = Klass(dirname=filename)
    else:
        raw_reader = Klass(filename=filename)

    return raw_reader
