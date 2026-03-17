# Instructions for AI Agents

## Overview
NDR-python is a faithful Python port of NDR-matlab (Neuroscience Data Reader).

## Architecture
- **Lead-Follow:** MATLAB is the source of truth. Python mirrors it exactly.
- **Bridge Contract:** Each sub-package has an `ndr_matlab_python_bridge.yaml`
  defining the function names, arguments, and return types.
- **Naming:** Preserve MATLAB names exactly. Use `readchannels_epochsamples`,
  not `read_channels_epoch_samples`.

## Key Classes
- `ndr.reader.base` — Abstract base class. All readers inherit from this.
- `ndr.reader` (wrapper) — High-level interface that delegates to a base reader.
- `ndr.reader.intan_rhd`, `ndr.reader.axon_abf`, etc. — Format-specific readers.

## Workflow
1. Check the bridge YAML in the target package.
2. If the function is missing, add it based on the MATLAB source.
3. Record the MATLAB git hash in `matlab_last_sync_hash`.
4. Implement the Python code.
5. Run `black` and `ruff check --fix` before committing.
6. Run `pytest` to verify.

## Testing
- Unit tests: `pytest tests/`
- Symmetry tests: `pytest tests/symmetry/` (excluded from default run)

## Environment
- Python 3.10+
- NumPy for all numerical data
- Pydantic for input validation (`@validate_call`)
