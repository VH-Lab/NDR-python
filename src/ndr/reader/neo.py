"""Neo reader class.

Port of +ndr/+reader/neo.m
Wraps the Python neo library to provide NDR-compatible interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ndr.reader.base import Base


class NeoReader(Base):
    """Reader that wraps the Python neo library.

    Port of ndr.reader.neo. Provides NDR interface for any format
    supported by the neo library.
    """

    def __init__(self) -> None:
        super().__init__()

    def daqchannels2internalchannels(
        self,
        channelprefix: list[str],
        channelstring: str | list[str],
        epochstreams: list[str],
        epoch_select: int = 1,
    ) -> list[dict[str, Any]]:
        """Convert channel names to internal structures.

        For Neo reader, channelstring is a list of native channel names.
        """
        try:
            import neo  # noqa: F401
        except ImportError as err:
            raise ImportError("neo is required for the Neo reader.") from err

        if isinstance(channelstring, str):
            channelstring = [channelstring]

        channelstruct: list[dict[str, Any]] = []
        for name in channelstring:
            channelstruct.append(
                {
                    "internal_type": "analog_in",
                    "internal_number": 0,
                    "internal_channelname": name,
                    "ndr_type": "analog_in",
                    "samplerate": float("nan"),
                }
            )
        return channelstruct

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
        raise NotImplementedError("Neo reader: use the specific neo IO classes directly.")

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read events."""
        return np.array([]), np.array([])
