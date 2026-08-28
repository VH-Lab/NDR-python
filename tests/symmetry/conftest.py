"""Shared fixtures and configuration for NDR symmetry tests."""

import tempfile
from pathlib import Path

from ndr.fun.ndrpath import ndrpath

# Base directory where all symmetry artifacts live:
#   <tempdir>/NDR/symmetryTest/<sourceType>/<namespace>/<class>/<test>/
SYMMETRY_BASE = Path(tempfile.gettempdir()) / "NDR" / "symmetryTest"
PYTHON_ARTIFACTS = SYMMETRY_BASE / "pythonArtifacts"
MATLAB_ARTIFACTS = SYMMETRY_BASE / "matlabArtifacts"

SOURCE_TYPES = ["matlabArtifacts", "pythonArtifacts"]

# The checked-in example data, resolved the same way MATLAB does it
# (`fullfile(ndr.fun.ndrpath(), 'example_data')`). Deriving this from
# ndrpath() rather than from __file__ keeps it correct regardless of where
# the tests are run from or whether the package is installed in place.
EXAMPLE_DATA = ndrpath() / "example_data"


def json_safe(value):
    """Convert a Python value into something both ports encode identically.

    MATLAB's ``jsonencode(..., 'ConvertInfAndNaN', true)`` writes NaN and Inf
    as ``null``; Python's ``json.dumps`` would emit bare ``NaN``/``Infinity``,
    which is not valid JSON and would not round-trip through MATLAB's
    ``jsondecode``. Numpy scalars and arrays are converted to plain Python so
    the encodings match element for element.

    2-D arrays become nested row lists, which is what MATLAB's ``jsonencode``
    produces for a matrix, so image planes compare correctly despite MATLAB
    being column-major and numpy row-major.
    """
    import math

    import numpy as np

    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
