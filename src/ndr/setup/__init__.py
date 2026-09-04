"""Setup helpers for optional NDR components.

Port of the ``+ndr/+setup`` package. Each module here installs or configures a
binding NDR cannot pull in through pyproject alone -- typically because its
own installer needs a private venv, a specific Python, or a vendor binary
that is not on PyPI.
"""
