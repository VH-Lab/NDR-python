"""TDT SEV reader class.

Port of +ndr/+reader/tdt_sev.m
Reads Tucker-Davis Technologies (.sev) files using the tdt library.
"""

from __future__ import annotations

import numpy as np

from ndr.reader.base import Base


class TdtSev(Base):
    """Reader for TDT .sev file format.

    Port of ndr.reader.tdt_sev.
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
        raise NotImplementedError("TDT reader not yet fully implemented.")

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
