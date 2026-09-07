"""OME-Zarr (NGFF v0.4) format reader.

Mirrors ``+ndr/+format/+omezarr/`` in NDR-matlab.
"""

from ndr.format.omezarr.isOMEZarr import isOMEZarr
from ndr.format.omezarr.listPyramids import listPyramids
from ndr.format.omezarr.readArray import readArray
from ndr.format.omezarr.readAttrs import readAttrs
from ndr.format.omezarr.resolveArrayPath import resolveArrayPath

__all__ = [
    "isOMEZarr",
    "listPyramids",
    "readArray",
    "readAttrs",
    "resolveArrayPath",
]
