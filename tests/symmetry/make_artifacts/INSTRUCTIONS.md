# NDR Symmetry Artifacts Instructions (Python — make_artifacts)

This folder contains Python tests whose purpose is to generate standard NDR artifacts for symmetry testing with other NDR language ports (e.g., MATLAB).

## Rules for `make_artifacts` tests:

1. **Artifact Location**: Tests must store their generated artifacts in the system's temporary directory (`tempfile.gettempdir()`).
2. **Directory Structure**: Inside the temporary directory, artifacts must be placed in a specific nested folder structure:
   `NDR/symmetryTest/pythonArtifacts/<namespace>/<class_name>/<test_name>/`

   - `<namespace>`: The sub-package name under `make_artifacts`. For example, for a test located at `tests/symmetry/make_artifacts/reader/`, the namespace is `reader`.
   - `<class_name>`: The name of the test class (e.g., `readData`), written in camelCase to match MATLAB conventions.
   - `<test_name>`: The specific name of the test method being executed (e.g., `testReadDataArtifacts`), also in camelCase.

3. **Persistent Teardown**: The generated artifact files **must persist** in the temporary directory so that the MATLAB test suite can read them. Do **not** use `tmp_path` for the artifact output directory — only use it for intermediate scratch state that can be discarded.

4. **Artifact Contents**: Every `make_artifacts` test should produce at minimum:
   - A `metadata.json` file describing the channels, sample rates, `t0`/`t1` boundaries,
     and epoch clock types returned by the reader.
   - A `readData.json` file containing a small, reproducible sample of data read via
     `readchannels_epochsamples(...)` so the MATLAB suite can verify numerical parity.

5. **Deterministic Input**: Tests should read from the checked-in example data files shared
   between NDR-matlab and NDR-python so that both language ports operate on byte-identical
   inputs.

6. **Imports**: Use the shared constant `PYTHON_ARTIFACTS` from `tests/symmetry/conftest.py`
   to build the artifact path. The base directory is:
   `<tempdir>/NDR/symmetryTest/pythonArtifacts/`.

## Example:

For a test class `TestReadData` in `tests/symmetry/make_artifacts/reader/test_read_data.py`
with a test method `test_read_data_artifacts`, the artifacts should be saved to:

```
<tempdir>/NDR/symmetryTest/pythonArtifacts/reader/readData/testReadDataArtifacts/
```

## Running

```bash
# Generate artifacts
pytest tests/symmetry/make_artifacts/ -v
```

## Adding a new symmetry test:

1. Create a sub-package under `make_artifacts/` named after the NDR domain (e.g., `reader/`,
   `format/`, `time/`).
2. Add a `test_<name>.py` file with a test class that exercises the reader (or other API)
   and writes `metadata.json` and `readData.json` (plus any other JSON blobs you need) to
   the artifact path described above.
3. Mirror the directory naming in MATLAB:
   `tools/tests/+ndr/+symmetry/+makeArtifacts/+<namespace>/<ClassName>.m`.
4. Add a corresponding `read_artifacts` test that can verify the generated artifacts (see
   `tests/symmetry/read_artifacts/INSTRUCTIONS.md`).
