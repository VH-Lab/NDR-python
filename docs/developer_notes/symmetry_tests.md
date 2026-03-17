# Cross-Language Symmetry Test Framework

**Status:** Active
**Scope:** NDR-python <-> NDR-matlab parity

## Purpose
Symmetry tests verify that data read by one language implementation matches
the other. This ensures the Python and MATLAB NDR stacks remain interoperable.

## Architecture

| Phase | Python location | MATLAB location |
|-------|----------------|-----------------|
| **makeArtifacts** | `tests/symmetry/make_artifacts/` | `tests/+ndr/+symmetry/+makeArtifacts/` |
| **readArtifacts** | `tests/symmetry/read_artifacts/` | `tests/+ndr/+symmetry/+readArtifacts/` |

### Artifact Directory Layout

```
<tempdir>/NDR/symmetryTest/
├── pythonArtifacts/
│   └── <namespace>/<className>/<testName>/
│       ├── readData.json           # Channel data, timestamps, etc.
│       └── metadata.json           # Channel list, sample rates, epoch info
└── matlabArtifacts/
    └── <namespace>/<className>/<testName>/
        └── ... (same structure)
```

### Workflow

1. **makeArtifacts** (Python or MATLAB) reads example data files and writes
   JSON artifacts containing: channel lists, sample rates, epoch clocks,
   t0/t1 boundaries, and actual data samples.

2. **readArtifacts** (the other language) loads the same example data files,
   reads the same channels/epochs, and compares against the stored artifacts.

Each `readArtifacts` test is parameterized over `{matlabArtifacts, pythonArtifacts}`.

## Running

```bash
# Generate artifacts
pytest tests/symmetry/make_artifacts/ -v

# Verify artifacts
pytest tests/symmetry/read_artifacts/ -v
```

## Writing a New Symmetry Test
See NDI-python's `docs/developer_notes/symmetry_tests.md` for the full template.
Adapt the pattern for NDR's reader-centric API.
