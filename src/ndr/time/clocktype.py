"""NDR clock type definitions.

Port of +ndr/+time/clocktype.m
"""

from __future__ import annotations

VALID_CLOCK_TYPES = (
    "utc",
    "exp_global_time",
    "dev_global_time",
    "dev_local_time",
    "approx_dev_local_time",
    "dev_global_time_no_tz",
    "exp_global_time_no_tz",
    "no_time",
    "inherited",
)


class ClockType:
    """Represents a timing specification for neural data epochs.

    Port of ndr.time.clocktype.
    """

    def __init__(self, clock_type: str = "dev_local_time"):
        self.type: str = ""
        self.setclocktype(clock_type)

    def setclocktype(self, clock_type: str) -> None:
        """Set the clock type, validating against known types."""
        if clock_type not in VALID_CLOCK_TYPES:
            raise ValueError(
                f"Unknown clock type '{clock_type}'. "
                f"Valid types are: {', '.join(VALID_CLOCK_TYPES)}"
            )
        self.type = clock_type

    def needsepoch(self) -> bool:
        """Return True if this clock type requires an epoch reference."""
        return self.type == "dev_local_time"

    def ndr_clocktype2char(self) -> str:
        """Return the clock type as a string."""
        return self.type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClockType):
            return NotImplemented
        return self.type == other.type

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, ClockType):
            return NotImplemented
        return self.type != other.type

    def __repr__(self) -> str:
        return f"ClockType('{self.type}')"

    def __str__(self) -> str:
        return self.type
