"""Regression test: `import ndr` must not require the optional tifffile dep.

reader/__init__.py imports tiffstack, which used to do a module-level
`import tifffile`, so `import ndr` raised ModuleNotFoundError in any environment
without tifffile -- including the pip install 'ndr[formats]' NDI-python declares.
The tifffile import is now lazy: `import ndr` and `import ndr.reader` succeed and
constructing the tiffstack reader succeeds; only actually using it raises.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _purge_ndr(monkeypatch):
    for name in [m for m in list(sys.modules) if m == "ndr" or m.startswith("ndr.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_import_ndr_without_tifffile(monkeypatch):
    # Setting the module to None makes `import tifffile` raise, simulating an
    # environment where tifffile is not installed.
    monkeypatch.setitem(sys.modules, "tifffile", None)
    _purge_ndr(monkeypatch)

    importlib.import_module("ndr")
    importlib.import_module("ndr.reader")


def test_tiffstack_constructs_but_use_raises_without_tifffile(monkeypatch):
    monkeypatch.setitem(sys.modules, "tifffile", None)
    _purge_ndr(monkeypatch)

    tiffstack = importlib.import_module("ndr.reader.tiffstack")
    reader = tiffstack.ndr_reader_tiffstack()  # construction is lazy -> OK

    # Any method that needs tifffile raises a clear ImportError.
    with pytest.raises(ImportError):
        reader.resolveepoch(["/nonexistent/file.tif"])
