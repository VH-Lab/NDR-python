# NDR MATLAB to Python Porting Guide

## 1. The Core Philosophy: Lead-Follow Architecture
The MATLAB codebase is the **Source of Truth**. The Python version is a "faithful mirror."
When a conflict arises between "Pythonic" style and MATLAB symmetry, **symmetry wins**.

- **Lead-Follow:** MATLAB defines the logic, hierarchy, and naming.
- **The Contract:** Every package contains an `ndr_matlab_python_bridge.yaml`.
  This file is the binding contract for function names, arguments, and return types.

## 2. Naming & Discovery (The Mirror Rule)
Function and class names must match MATLAB exactly.

- **Naming Source:** Refer to the local `ndr_matlab_python_bridge.yaml`.
- **Missing Entries:** If a function is not in the bridge file, refer to the MATLAB
  source, add the entry to the bridge file, and notify the user.
- **Case Preservation:** Use `readchannels_epochsamples`, not `read_channels_epoch_samples`.
- **Directory Parity:** Python file paths must mirror MATLAB `+namespace` paths
  (e.g., `+ndr/+reader` -> `src/ndr/reader/`).

## 3. The Porting Workflow (The Bridge Protocol)
1. **Check the Bridge:** Open the `ndr_matlab_python_bridge.yaml` in the target package.
2. **Sync the Interface:** If the function is missing or outdated, update the YAML first.
3. **Record the Sync Hash:** Store the short git hash of the MATLAB `.m` file:
   `git log -1 --format="%h" -- <path-to-matlab-file>`
4. **Implement:** Write Python code to satisfy the interface defined in the YAML.
5. **Log & Notify:** Record the sync date in the YAML's `decision_log`.

## 4. Input Validation: Pydantic is Mandatory
Use `@pydantic.validate_call` on all public-facing API functions.

- MATLAB `double`/`numeric` -> Python `float | int`
- MATLAB `char`/`string` -> Python `str`
- MATLAB `{member1, member2}` -> Python `Literal["member1", "member2"]`

## 5. Multiple Returns (Outputs)
Return multiple values as a **tuple** in the exact order defined in the YAML.

## 6. Code Style & Linting
- **Black:** The sole code formatter. Line length 100.
- **Ruff:** The primary linter. Run `ruff check --fix` before committing.

## 7. Error Handling
If MATLAB throws an `error`, Python MUST raise a corresponding Exception.
