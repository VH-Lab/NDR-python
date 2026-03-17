"""Shared fixtures and configuration for NDR symmetry tests."""

import tempfile
from pathlib import Path

# Base directory where all symmetry artifacts live:
#   <tempdir>/NDR/symmetryTest/<sourceType>/<namespace>/<class>/<test>/
SYMMETRY_BASE = Path(tempfile.gettempdir()) / "NDR" / "symmetryTest"
PYTHON_ARTIFACTS = SYMMETRY_BASE / "pythonArtifacts"
MATLAB_ARTIFACTS = SYMMETRY_BASE / "matlabArtifacts"

SOURCE_TYPES = ["matlabArtifacts", "pythonArtifacts"]
