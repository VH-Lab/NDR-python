"""A class for specifying time relative to an NDR clock.

Port of +ndr/+time/timereference.m
"""

from __future__ import annotations

from typing import Any

from ndr.time.clocktype import ClockType


class TimeReference:
    """Specifies a time relative to an NDR clock and referent.

    A time reference object ties together a referent (an epochset subclass),
    a clock type, an optional epoch identifier, a time value, and a session ID
    to unambiguously specify a point in time within an NDR session.

    Parameters
    ----------
    referent : object
        An epochset subclass object that has a ``session`` property
        (e.g., an ndr.system, ndr.element, etc.).  When using the
        struct-based constructor form, pass the session object here.
    clocktype : ClockType or dict
        The clock type specification.  When using the struct-based
        constructor, pass a ``dict`` returned by
        :meth:`ndr_timereference_struct` here.
    epoch : int, str, or None
        The epoch identifier.  Required when *clocktype* is
        ``'dev_local_time'``; may be ``None`` otherwise.
    time : float or None
        The time value relative to the referent / epoch.

    Raises
    ------
    TypeError
        If *referent* does not expose a ``session`` property, or if
        *clocktype* is not a :class:`~ndr.time.clocktype.ClockType`.
    ValueError
        If *clocktype* requires an epoch but none is provided.

    Notes
    -----
    An alternative constructor form is supported::

        TimeReference(session_obj, timeref_struct)

    where *session_obj* is an ``ndr.session`` and *timeref_struct* is a
    ``dict`` previously returned by :meth:`ndr_timereference_struct`.
    The session's ``findexpobj`` method will be called to resolve the
    live referent.
    """

    def __init__(
        self,
        referent: Any,
        clocktype: ClockType | dict[str, Any],
        epoch: int | str | None = None,
        time: float | None = None,
    ) -> None:
        session_ID: str

        # ------------------------------------------------------------------
        # Alternate two-argument form: (session, timeref_struct)
        # ------------------------------------------------------------------
        if isinstance(clocktype, dict):
            session = referent
            timeref_struct = clocktype
            session_ID = session.id()
            referent = session.findexpobj(
                timeref_struct["referent_epochsetname"],
                timeref_struct["referent_classname"],
            )
            clocktype = ClockType(timeref_struct["clocktypestring"])
            epoch = timeref_struct.get("epoch")
            time = timeref_struct.get("time")

        # ------------------------------------------------------------------
        # Validate referent
        # ------------------------------------------------------------------
        if not hasattr(referent, "epochsetname"):
            raise TypeError(
                "referent must be a subclass of ndr.epoch.epochset "
                "(must have an 'epochsetname' method)."
            )

        if hasattr(referent, "session"):
            session_obj = (
                referent.session()
                if callable(referent.session)
                else referent.session
            )
            if session_obj is None:
                raise TypeError(
                    "The referent must have a session with a valid id."
                )
            session_ID = session_obj.id() if callable(getattr(session_obj, "id", None)) else str(session_obj)
        else:
            raise TypeError(
                "The referent must have a session property with a valid id."
            )

        # ------------------------------------------------------------------
        # Validate clocktype
        # ------------------------------------------------------------------
        if not isinstance(clocktype, ClockType):
            raise TypeError(
                "clocktype must be an instance of ndr.time.clocktype.ClockType."
            )

        # ------------------------------------------------------------------
        # Validate epoch requirement
        # ------------------------------------------------------------------
        if clocktype.needsepoch():
            if epoch is None:
                raise ValueError(
                    "Time is local; an epoch must be specified."
                )

        # ------------------------------------------------------------------
        # Assign properties
        # ------------------------------------------------------------------
        self._referent = referent
        self._session_ID: str = session_ID
        self._clocktype: ClockType = clocktype
        self._epoch: int | str | None = epoch
        self._time: float | None = time

    # ------------------------------------------------------------------
    # Read-only properties (mirrors MATLAB SetAccess=protected)
    # ------------------------------------------------------------------

    @property
    def referent(self) -> Any:
        """The epochset subclass object that is referred to."""
        return self._referent

    @property
    def clocktype(self) -> ClockType:
        """The :class:`~ndr.time.clocktype.ClockType` of this reference."""
        return self._clocktype

    @property
    def epoch(self) -> int | str | None:
        """The epoch identifier (required for ``'dev_local_time'``)."""
        return self._epoch

    @property
    def time(self) -> float | None:
        """The time value of the referent."""
        return self._time

    @property
    def session_ID(self) -> str:
        """The session ID that contains this time reference."""
        return self._session_ID

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def ndr_timereference_struct(self) -> dict[str, Any]:
        """Return a plain-dict description of this time reference.

        The returned dictionary contains no live objects and is suitable
        for serialisation or reconstruction via the two-argument
        constructor form.

        Returns
        -------
        dict
            Keys:

            * ``referent_epochsetname`` -- the epochset name of the referent
            * ``referent_classname`` -- the class name of the referent
            * ``clocktypestring`` -- the clock type as a string
            * ``epoch`` -- the epoch (string or int)
            * ``session_ID`` -- the session identifier
            * ``time`` -- the time value
        """
        epochsetname = self._referent.epochsetname()
        classname = type(self._referent).__name__

        return {
            "referent_epochsetname": epochsetname,
            "referent_classname": classname,
            "clocktypestring": self._clocktype.ndr_clocktype2char(),
            "epoch": self._epoch,
            "session_ID": self._session_ID,
            "time": self._time,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TimeReference(referent={self._referent!r}, "
            f"clocktype={self._clocktype!r}, "
            f"epoch={self._epoch!r}, time={self._time!r})"
        )
