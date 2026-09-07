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
2. If the function is missing, add it based on the MATLAB source with
   `status: regular_port`. If it won't be ported 1:1, add an entry with the
   matching `status` and a `decision_log` explaining why — the CI
   completeness check fails on unrecorded `.m` files, and the conventions
   check fails on an entry with no `status` at all.
3. Record the MATLAB commit in `matlab_last_sync_hash`.
4. Implement the Python code.
5. Run `black` and `ruff check --fix` before committing.
6. Run `pytest` to verify.

**The bridge rules live in exactly one place:**
[`docs/developer_notes/ndr_matlab_python_bridge.yaml`](docs/developer_notes/ndr_matlab_python_bridge.yaml).
Do not restate them here or anywhere else — a rule written in three
documents is a rule that will eventually contradict itself, which is how
NDI-python ended up with a spec that asked for a blob hash in prose and a
commit hash in the command printed beneath it. Go there for:

- **Section 4** — one entry per MATLAB function, and the one declared
  exception.
- **Section 5** — `matlab_last_sync_hash` is a COMMIT (`git -C
  ../NDR-matlab log -n 1 --format=%h -- <path>`), never a blob, and it must
  be the latest commit touching the file.
- **Section 6** — the complete `status` vocabulary: `regular_port`,
  `ported_differently`, `porting_deferred`, `matlab_only`, `retired`. EVERY
  entry states its status, including the ordinary one: a plain 1:1 port is
  written `status: regular_port`, never left blank, so a human reading the
  YAML can tell a finished entry from an unfilled one. Every status except
  `regular_port` also needs a `decision_log`. `ported_elsewhere`,
  `not_yet_ported`, `not_applicable`, `ported` and `implemented` are retired
  spellings and CI rejects them.

## Testing
- Unit tests: `pytest tests/`
- Symmetry tests: `pytest tests/symmetry/` (excluded from default run)
- Bridge checks — completeness, hash currency and conventions:
  ```
  NDR_BRIDGE_CHECK_STRICT=1 NDR_MATLAB_PATH=../NDR-matlab \
    pytest tests/test_matlab_bridge_completeness.py \
           tests/test_matlab_bridge_conventions.py
  ```
  Requires a NON-shallow NDR-matlab checkout (a shallow clone collapses
  file history to the last fetched commit and lies about which hash is
  latest); if you cloned with `--depth`, run `git fetch --unshallow`
  first. CI does this via `fetch-depth: 0`, and the tests now fail with
  one clear message on a shallow clone instead of one false failure per
  entry.

  A new `tests/test_matlab_bridge_*.py` must be added to the `bridge` job
  in `.github/workflows/ci.yml` by name. The test matrix has no NDR-matlab
  checkout, so an unwired bridge test skips everywhere and the skip reads
  as a pass; `test_matlab_bridge_conventions.py` fails if you forget.

## Environment
- Python 3.10+
- NumPy for all numerical data
- Pydantic for input validation (`@validate_call`)
