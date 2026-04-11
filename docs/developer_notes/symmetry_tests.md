# Cross-Language Symmetry Test Framework

**Status:** Active
**Scope:** NDR-python ↔ NDR-matlab parity

## Purpose

Symmetry tests verify that data read by one language implementation matches the
other. This ensures the Python and MATLAB NDR stacks remain interoperable as
both codebases evolve, and mirrors the same framework that NDI-python and
NDI-matlab already use.

## Architecture

The framework has two halves, each existing in both languages:

| Phase | Python location | MATLAB location |
|-------|----------------|-----------------|
| **makeArtifacts** | `tests/symmetry/make_artifacts/` | `tools/tests/+ndr/+symmetry/+makeArtifacts/` |
| **readArtifacts** | `tests/symmetry/read_artifacts/` | `tools/tests/+ndr/+symmetry/+readArtifacts/` |

### Artifact Directory Layout

All artifacts are written to the OS temporary directory under a fixed path:

```
<tempdir>/NDR/symmetryTest/
├── pythonArtifacts/
│   └── <namespace>/<className>/<testName>/
│       ├── metadata.json     # Channel list, sample rates, t0/t1, epoch clocks
│       └── readData.json     # Short reproducible sample of reader output
└── matlabArtifacts/
    └── <namespace>/<className>/<testName>/
        └── ... (same structure)
```

- **`<namespace>`** — the NDR domain being tested (e.g., `reader`).
- **`<className>`** — the test class name, in camelCase (e.g., `readData`).
- **`<testName>`** — the test method name, in camelCase (e.g., `testReadDataArtifacts`).

### Workflow

```
┌──────────────────────────┐     ┌──────────────────────────┐
│  Python makeArtifacts    │     │  MATLAB makeArtifacts    │
│  pytest tests/symmetry/  │     │  runtests('ndr.symmetry. │
│    make_artifacts/ -v    │     │    makeArtifacts')       │
└──────────┬───────────────┘     └──────────┬───────────────┘
           │ writes                          │ writes
           ▼                                 ▼
     pythonArtifacts/                  matlabArtifacts/
           │                                 │
           └────────────┬────────────────────┘
                        │ reads
           ┌────────────┴────────────────────┐
           │                                 │
           ▼                                 ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│  Python readArtifacts    │     │  MATLAB readArtifacts    │
│  pytest tests/symmetry/  │     │  runtests('ndr.symmetry. │
│    read_artifacts/ -v    │     │    readArtifacts')       │
└──────────────────────────┘     └──────────────────────────┘
```

Each `readArtifacts` test is parameterized over `{matlabArtifacts, pythonArtifacts}`
so a single test class validates both directions of compatibility.

## Running the Tests

### From Python

```bash
# Generate artifacts
pytest tests/symmetry/make_artifacts/ -v

# Verify artifacts (skips missing sources)
pytest tests/symmetry/read_artifacts/ -v

# Both phases at once
pytest tests/symmetry/ -v
```

### From MATLAB

```matlab
% Generate artifacts
results = runtests('ndr.symmetry.makeArtifacts', 'IncludeSubpackages', true);

% Verify artifacts
results = runtests('ndr.symmetry.readArtifacts', 'IncludeSubpackages', true);
```

### Why Separate from Regular Tests?

Symmetry tests are **excluded from the default `pytest` run** (via
`--ignore=tests/symmetry` in `pyproject.toml`) because:

1. **readArtifacts** tests will mostly just skip unless the user has previously
   run MATLAB's `makeArtifacts` suite on the same machine.
2. **makeArtifacts** tests write to the system temp directory, which is a
   side-effect that doesn't belong in routine CI.
3. The full cross-language cycle requires both runtimes and is better suited to
   integration / nightly CI pipelines.

## Writing a New Symmetry Test

### 1. Choose a namespace

Pick the NDR domain being tested (e.g., `reader`, `format`, `time`).

### 2. Create the makeArtifacts test

**Python:** `tests/symmetry/make_artifacts/<namespace>/test_<name>.py`

```python
import json
import shutil
from pathlib import Path
import pytest

from tests.symmetry.conftest import PYTHON_ARTIFACTS

ARTIFACT_DIR = PYTHON_ARTIFACTS / "<namespace>" / "<className>" / "<testName>"
EXAMPLE_DATA = Path(__file__).parents[4] / "example_data"


class TestMyFeature:
    def test_my_feature_artifacts(self):
        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        # ... build reader, collect metadata, dump metadata.json + readData.json
```

**MATLAB:** `tools/tests/+ndr/+symmetry/+makeArtifacts/+<namespace>/<ClassName>.m`

Follow the `INSTRUCTIONS.md` in the MATLAB `+makeArtifacts` folder.

### 3. Create the readArtifacts test

**Python:** `tests/symmetry/read_artifacts/<namespace>/test_<name>.py`

```python
import json
import pytest
from tests.symmetry.conftest import SOURCE_TYPES, SYMMETRY_BASE


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


class TestMyFeature:
    def test_my_feature_artifacts(self, source_type):
        artifact_dir = SYMMETRY_BASE / source_type / "<namespace>" / "<className>" / "<testName>"
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")
        # ... load metadata.json / readData.json and assert against re-read values
```

**MATLAB:** `tools/tests/+ndr/+symmetry/+readArtifacts/+<namespace>/<ClassName>.m`

Follow the `INSTRUCTIONS.md` in the MATLAB `+readArtifacts` folder.

### 4. Naming Conventions

| Concept | Python | MATLAB |
|---------|--------|--------|
| Test directory | `tests/symmetry/make_artifacts/reader/` | `tools/tests/+ndr/+symmetry/+makeArtifacts/+reader/` |
| Test file | `test_read_data.py` | `readData.m` |
| Test class | `TestReadData` | `readData` (classdef) |
| Artifact className | `readData` (camelCase) | `readData` |
| Artifact testName | `testReadDataArtifacts` (camelCase) | `testReadDataArtifacts` |

Use **camelCase** for the artifact directory components (`className`, `testName`)
so that both languages write to and read from the exact same paths.

## Related: Building the Bridge YAML Files

Symmetry tests confirm **behavioural** parity at runtime. The bridge YAML files
under `src/ndr/*/ndr_matlab_python_bridge.yaml` describe **interface** parity at
the signature level. See
`docs/developer_notes/ndr_matlab_python_bridge.yaml` for the full spec, and
Section 5 of that file ("Building a New Bridge YAML File") for the step-by-step
procedure to add or update a contract. The `PYTHON_PORTING_GUIDE.md` also
documents the porting workflow, including how to compute and record
`matlab_last_sync_hash`.
