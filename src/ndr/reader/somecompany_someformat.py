"""Template reader for a particular company/group's data format.

Port of +ndr/+reader/somecompany_someformat.m
"""

from __future__ import annotations

import numpy as np

from ndr.reader.base import ndr_reader_base


class ndr_reader_somecompany__someformat(ndr_reader_base):
    """NDR reader template for a particular company or group's format.

    This is a template class for making :class:`ndr.reader.base.ndr_reader_base`
    subclasses that interpret and read data from a particular company
    or group (SOMECOMPANY) in a particular format (SOMEFORMAT).

    Subclass this template and implement the abstract methods from
    :class:`~ndr.reader.base.ndr_reader_base` to create a working reader.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Abstract method stubs (must be implemented by concrete readers)
    # ------------------------------------------------------------------

    def readchannels_epochsamples(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        s0: int,
        s1: int,
    ) -> np.ndarray:
        """Read data from specified channels.

        Parameters
        ----------
        channeltype : str
            Type of channel to read.
        channel : int or list of int
            Channel number(s) to read (1-based).
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        s0 : int
            Start sample number (1-based).
        s1 : int
            End sample number (1-based).

        Returns
        -------
        numpy.ndarray
            Data array with one column per channel.
        """
        raise NotImplementedError(
            "readchannels_epochsamples must be implemented by a concrete reader subclass."
        )

    def readevents_epochsamples_native(
        self,
        channeltype: str,
        channel: int | list[int],
        epochstreams: list[str],
        epoch_select: int,
        t0: float,
        t1: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read events or markers for specified channels.

        Parameters
        ----------
        channeltype : str
            Type of channel to read.
        channel : int or list of int
            Channel number(s) to read.
        epochstreams : list of str
            File paths comprising the epoch.
        epoch_select : int
            Which epoch to access.
        t0 : float
            Start time.
        t1 : float
            End time.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            (timestamps, data)
        """
        raise NotImplementedError(
            "readevents_epochsamples_native must be implemented by a concrete reader subclass."
        )
