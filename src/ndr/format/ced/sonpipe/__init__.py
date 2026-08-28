"""Read CED Spike2 files through the sonpipe command-line bridge.

Port of the +ndr/+format/+ced/+sonpipe package. See executable.py for why the
reader runs out of process rather than importing sonpy directly.
"""

from ndr.format.ced.sonpipe._invoke import SonpipeError, invoke_binary, invoke_json, invoke_text
from ndr.format.ced.sonpipe.channelinfo import channelinfo
from ndr.format.ced.sonpipe.executable import SonpipeNotFoundError, executable, reset_cache
from ndr.format.ced.sonpipe.read_SOMSMR_datafile import read_SOMSMR_datafile
from ndr.format.ced.sonpipe.read_SOMSMR_header import read_SOMSMR_header
from ndr.format.ced.sonpipe.read_SOMSMR_sampleinterval import read_SOMSMR_sampleinterval

__all__ = [
    "SonpipeError",
    "SonpipeNotFoundError",
    "channelinfo",
    "executable",
    "invoke_binary",
    "invoke_json",
    "invoke_text",
    "read_SOMSMR_datafile",
    "read_SOMSMR_header",
    "read_SOMSMR_sampleinterval",
    "reset_cache",
]
