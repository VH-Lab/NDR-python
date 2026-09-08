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
- **Section 5 / 5a** — `matlab_last_sync_hash` is a COMMIT (`git -C
  ../NDR-matlab log -n 1 --format=%h -- <path>`), never a blob. Either the
  file's own last-touching commit or a repo-wide batch-sync commit is legal.
  What fails CI is DRIFT: `git log <hash>..HEAD -- <matlab_path>` non-empty,
  i.e. NDR-matlab has touched that file since. File-scoped — unrelated repo
  activity never trips it. Every entry naming a `matlab_path` must carry a
  hash — one that is absent can never drift, so it would claim to be current
  forever. Rules set for all three bridge repos by NDI-python#211.
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

### A skip is a silent pass — a note for whoever works on these next

The bridge guard exists because *a check that could not run must not report
the same result as a check that ran and passed*. The subtle part is that this
applies to the guard's own tests, and it is easy to violate while writing
something careful. It has now happened three times in this repo:

1. The `bridge` job named one test file, so a new one would have skipped in
   every job and read as green.
2. MATLAB-dependent tests **skipped** in the three matrix jobs — 79 each —
   until they were deselected instead. A standing pile of skips trains a
   reader to scroll past skips, so the next one, a real one, goes unnoticed.
3. The drift self-tests — the positive controls proving the drift rule
   actually discriminates — carried three `pytest.skip` calls gated on
   nothing. Had NDR-matlab's history made their fixture conditions true,
   the proof that the rule works would have stopped running, silently.

None of these was careless in isolation; each was a reasonable local decision
(*a test that cannot build its fixture should not fail*) that became a hole in
CI. So the rule here is not "never skip" — it is:

> **A `pytest.skip` in a bridge test must be conditional on
> `NDR_BRIDGE_CHECK_STRICT`, or it will hide.** In CI every bridge check is
> expected to be *able* to run, so a skip there is a gap, not a courtesy.

`tests/conftest.py` now enforces this as a backstop: under
`NDR_BRIDGE_CHECK_STRICT`, **any** skip in a `test_matlab_bridge_*.py` module
becomes a failure — including one nobody thought to gate. It is derived rather
than enumerated for the same reason the `needs_matlab` marker is: a list of
exceptions is a list somebody has to remember to extend.

If it fires on you, the fix is to make the condition hold in CI, or to gate
the skip yourself with a message saying why the check could not run. Deleting
the assertion to get green is the one thing that is never the answer — that is
precisely the move that put the Intan multi-file feature behind a green build
for four months (see issues #20 and #23).

## Environment
- Python 3.10+
- NumPy for all numerical data
- Pydantic for input validation (`@validate_call`)
