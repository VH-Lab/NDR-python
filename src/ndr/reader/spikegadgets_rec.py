"""SpikeGadgets REC reader class.

Port of +ndr/+reader/spikegadgets_rec.m
"""

from __future__ import annotations

import numpy as np

from ndr.reader.base import Base


class SpikeGadgetsRec(Base):
    """Reader for SpikeGadgets .rec file format.

    Port of ndr.reader.spikegadgets_rec.
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
        raise NotImplementedError("SpikeGadgets reader not yet fully implemented.")

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
