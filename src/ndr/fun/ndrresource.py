"""NDR resource loading utilities.

Port of +ndr/+fun/ndrresource.m
"""

from __future__ import annotations

import json
from typing import Any

from ndr.fun.ndrpath import ndrpath


def ndrresource(resource_name: str) -> Any:
    """Load a JSON resource file from the NDR resource directory.

    Parameters
    ----------
    resource_name : str
        Name of the resource file (e.g., 'ndr_reader_types.json').

    Returns
    -------
    Any
        Parsed JSON content (typically a list or dict).
    """
    resource_dir = ndrpath() / "resource"
    filepath = resource_dir / resource_name

    if not filepath.exists():
        raise FileNotFoundError(f"NDR resource file '{resource_name}' not found at {filepath}.")

    text = filepath.read_text(encoding="utf-8")
    return json.loads(text)
