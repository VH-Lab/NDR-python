"""White Matter reader class.

Port of +ndr/+reader/whitematter.m
"""

from __future__ import annotations

import numpy as np

from ndr.reader.base import ndr_reader_base


class ndr_reader_whitematter(ndr_reader_base):
    """Reader for White Matter file format.

    Port of ndr.reader.whitematter.
    """

    def __init__(self) -> None:
        super().__init__()

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
        raise NotImplementedError("WhiteMatter reader not yet fully implemented.")

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
