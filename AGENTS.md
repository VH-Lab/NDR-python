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
2. If the function is missing, add it based on the MATLAB source. If it
   won't be ported, still add an entry with `status: not_yet_ported` or
   `not_applicable` and a `decision_log` explaining why — the CI
   completeness check fails on unrecorded `.m` files.
3. Record the MATLAB git hash in `matlab_last_sync_hash` — the SHORT
   hash of the latest commit that touched the MATLAB file (get it with
   `git -C ../NDR-matlab log -n 1 --format=%h -- <path>`). CI enforces
   that this is the current-latest hash, so when MATLAB edits the file
   you MUST either port the change and bump the hash, or bump the hash
   and add a short `decision_log` note saying why the MATLAB change is
   a no-op here (comment-only, analyzer fix, etc.).
4. Implement the Python code.
5. Run `black` and `ruff check --fix` before committing.
6. Run `pytest` to verify.

## Testing
- Unit tests: `pytest tests/`
- Symmetry tests: `pytest tests/symmetry/` (excluded from default run)
- Bridge completeness + hash-currency:
  `NDR_MATLAB_PATH=../NDR-matlab pytest tests/test_matlab_bridge_completeness.py`.
  Requires a NON-shallow NDR-matlab checkout (a shallow clone collapses
  file history to the last merge commit and lies about which hash is
  latest); if you cloned with `--depth`, run `git fetch --unshallow`
  first. CI does this via `fetch-depth: 0`.

## Environment
- Python 3.10+
- NumPy for all numerical data
- Pydantic for input validation (`@validate_call`)
