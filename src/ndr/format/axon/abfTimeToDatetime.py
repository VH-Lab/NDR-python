"""Convert ABF file time fields to Python datetime.

Port of +ndr/+format/+axon/abfTimeToDatetime.m
"""

from __future__ import annotations

from datetime import datetime


def abfTimeToDatetime(uFileStartDate: float, uFileStartTimeMS: float) -> datetime:
    """Convert ABF file time fields to a Python datetime.

    Parameters
    ----------
    uFileStartDate : float
        ABF file start date in YYYYMMDD format.
    uFileStartTimeMS : float
        ABF file start time in milliseconds since midnight.

    Returns
    -------
    datetime
        The date and time of the recording start.
    """
    if uFileStartTimeMS < 0 or uFileStartTimeMS >= 86400000:
        raise ValueError("uFileStartTimeMS must be between 0 and 86,399,999")

    date_str = f"{int(uFileStartDate):08d}"
    if len(date_str) != 8:
        raise ValueError("uFileStartDate must be in YYYYMMDD format")

    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    if month < 1 or month > 12 or day < 1 or day > 31:
        raise ValueError("Invalid date components in uFileStartDate")

    ms = int(uFileStartTimeMS)
    hours = ms // 3600000
    ms -= hours * 3600000
    minutes = ms // 60000
    ms -= minutes * 60000
    seconds = ms // 1000
    milliseconds = ms - seconds * 1000

    return datetime(year, month, day, hours, minutes, seconds, milliseconds * 1000)
