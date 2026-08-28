# NDR-python Port Status

Status of the MATLAB → Python port of [NDR-matlab](https://github.com/VH-Lab/NDR-matlab).

**Last synchronized with NDR-matlab at `ac13506`, on 2026-08-28.**

> `ac13506` is on the NDR-matlab branch
> `claude/ndr-python-ndi-matlab-sync-im3ci6`, not yet on `main`. Six bridge
> entries (the four `+spikegadgets/read_rec_*` files and the two
> `+vld/readvhlv*` files) record it, because their Python counterparts were
> changed in lockstep with it. If that branch is squash-merged the commit
> hash will change and those six entries will report drift; re-point them at
> the merge commit rather than re-reviewing the files.

Every entry in every `ndr_matlab_python_bridge.yaml` now carries a
`matlab_last_sync_hash`, so upstream drift can be detected mechanically: for
each entry, compare that field against
`git log -1 --format="%h" -- <matlab_path>` in an NDR-matlab checkout. A
difference means the MATLAB file moved since the Python side was last
examined. See `docs/developer_notes/ndr_matlab_python_bridge.yaml` § 3a.

## Naming Convention

Python class names are a mechanical mapping of the fully-qualified MATLAB class name,
applying the **Mirror Rule**:

1. Periods (`.`) are replaced with single underscores (`_`).
2. Existing underscores (`_`) in the MATLAB name are replaced with double underscores (`__`).

| MATLAB qualified name | Python module | Python class |
|---|---|---|
| `ndr.reader` | `ndr.reader_wrapper` | `ndr_reader` |
| `ndr.reader.base` | `ndr.reader.base` | `ndr_reader_base` |
| `ndr.reader.intan_rhd` | `ndr.reader.intan_rhd` | `ndr_reader_intan__rhd` |
| `ndr.reader.ced_smr` | `ndr.reader.ced_smr` | `ndr_reader_ced__smr` |
| `ndr.reader.axon_abf` | `ndr.reader.axon_abf` | `ndr_reader_axon__abf` |
| `ndr.reader.neo` | `ndr.reader.neo` | `ndr_reader_neo` |
| `ndr.reader.spikegadgets_rec` | `ndr.reader.spikegadgets_rec` | `ndr_reader_spikegadgets__rec` |
| `ndr.reader.tdt_sev` | `ndr.reader.tdt_sev` | `ndr_reader_tdt__sev` |
| `ndr.reader.bjg` | `ndr.reader.bjg` | `ndr_reader_bjg` |
| `ndr.reader.dabrowska` | `ndr.reader.dabrowska` | `ndr_reader_dabrowska` |
| `ndr.reader.whitematter` | `ndr.reader.whitematter` | `ndr_reader_whitematter` |
| `ndr.reader.neuropixelsGLX` | `ndr.reader.neuropixelsGLX` | `ndr_reader_neuropixelsGLX` |
| `ndr.reader.tiffstack` | `ndr.reader.tiffstack` | `ndr_reader_tiffstack` |
| `ndr.reader.prairieview` | `ndr.reader.prairieview` | `ndr_reader_prairieview` |
| `ndr.reader.vld` | `ndr.reader.vld` | `ndr_reader_vld` |
| `ndr.reader.somecompany_someformat` | `ndr.reader.somecompany_someformat` | `ndr_reader_somecompany__someformat` |

## Reader Status

| Reader | getchannelsepoch | t0\_t1 | samplerate | readchannels\_epochsamples | readevents\_epochsamples\_native | read | Tests |
|---|---|---|---|---|---|---|---|
| **ndr\_reader\_intan\_\_rhd** | Yes | Yes | Yes | Yes (single-file) | Stub (empty) | Yes | 6 pass |
| **ndr\_reader\_ced\_\_smr** | Yes | Yes | Yes | Yes | Yes | Yes (via base) | 14 pass |
| **ndr\_reader\_axon\_\_abf** | Yes | Yes | Yes | Yes | Stub (empty) | Yes (via base) | 6 pass |
| **ndr\_reader\_neo** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | 24 xfail |
| **ndr\_reader\_spikegadgets\_\_rec** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | xfail |
| **ndr\_reader\_tdt\_\_sev** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **ndr\_reader\_bjg** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **ndr\_reader\_dabrowska** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **ndr\_reader\_whitematter** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **ndr\_reader\_neuropixelsGLX** | Yes | Yes | Yes | Yes | Stub (empty) | Yes (via base) | 39 pass |
| **ndr\_reader\_vld** | Yes | Yes | Yes | Yes | Stub (empty) | Yes (via base) | 25 pass |

All readers also implement `channelLabelingConvention` (MATLAB b2e9d95,
3974d59), declaring how the reader names channels: `ced_smr`,
`spikegadgets_rec` and `tdt_sev` return `'physical'`, `neo` returns
`'native'`, and the rest inherit the base default of `'indexed'`.

**Legend:**
- **Yes** — Fully implemented and tested with example data
- **Stub (empty)** — Returns empty arrays / default values; no errors raised
- **NotImplementedError** — Raises an exception; not yet implemented
- **Stub** — Inherits base class default (empty list, `[[nan,nan]]`, etc.)

## Format Parsers

Low-level format parsers (under `ndr.format.*`) that read binary files:

| Format | Module | Status |
|---|---|---|
| Intan RHD | `ndr.format.intan` | Implemented (header + single-file data reader) |
| CED SMR/SON | `ndr.format.ced` | Implemented (via `neo` library) |
| Axon ABF | `ndr.format.axon` | Implemented (via `pyabf` library) |
| SpikeGadgets REC | `ndr.format.spikegadgets` | Implemented (config, analog, digital, trode) |
| TDT SEV | `ndr.format.tdt` | Implemented (header + channel reader) |
| BJG | `ndr.format.bjg` | Implemented (header + data reader) |
| Dabrowska | `ndr.format.dabrowska` | Implemented (header + data reader) |
| WhiteMatter | `ndr.format.whitematter` | Implemented (header + data reader) |
| Neo / Blackrock | `ndr.format.neo` | Implemented (utilities) |
| Binary Matrix | `ndr.format.binarymatrix` | Implemented |
| Text Signal | `ndr.format.textSignal` | Implemented |

Note: For SpikeGadgets, TDT, BJG, Dabrowska, and WhiteMatter, the format parsers are implemented but the reader classes have not yet been wired up to use them.

## Reader Wrapper

The top-level `ndr_reader` class (`reader_wrapper.py`) wraps any format-specific reader and adds:

| Feature | Status |
|---|---|
| `read()` convenience method | Implemented |
| `readevents_epochsamples()` with derived events (dep, den, dimp, dimn) | Implemented |
| Delegation to underlying reader | Implemented |

## External Dependencies

| Dependency | Used by | Purpose |
|---|---|---|
| `neo` | ndr\_reader\_ced\_\_smr, ndr\_reader\_neo | Read CED SMR/SON and Blackrock files |
| `pyabf` | ndr\_reader\_axon\_\_abf | Read Axon Binary Format files |
| `tifffile` | ndr\_reader\_tiffstack, ndr\_reader\_prairieview | Read TIFF image stacks |
| `numpy` | All readers | Array operations |

## Imaging Support and Remaining Gaps

NDR-matlab grew several subsystems after the Python port was last synced. The
imaging stack (frame API, `tiffstack`, `prairieview`) and the `vld` reader are
now ported; what remains outstanding is listed explicitly below so the gap is
stated rather than implied by absence.

### Image / frame-reading API — **ported**

`ndr.reader.base` and the `ndr.reader` wrapper implement the frame-oriented
API for image series, alongside the sample-oriented one. A reader that handles
images implements only the frame API; readers that do not inherit no-op
defaults.

| Method | Purpose | Status |
|---|---|---|
| `numframes` | Number of frames (timepoints x planes) in the epoch | Implemented |
| `framesize` | `[Y X C Z T]` extent, without reading pixels | Implemented |
| `dimensionorder` | Dimension model of the returned array (default `'YXCZT'`) | Implemented |
| `datatype` | Underlying pixel data type | Implemented |
| `frametimes` | Per-frame acquisition times | Implemented |
| `readframes` | Read frames, with `SelectC` / `SelectZ` channel and plane selection | Implemented |
| `metadata` | Standardized image-acquisition metadata (raster timing) | Implemented |

Static helpers `emptyimagemetadata` and `selectframeCZ` are ported too.
Frame indices and `SelectC`/`SelectZ` are 1-based, per the bridge's Semantic
Parity policy. `datatype` holds **strict string parity** with MATLAB: both
ports return the MATLAB numeric class name, so `'single'`/`'double'` rather
than numpy's `'float32'`/`'float64'`, and the integer names match already.
Python callers that need the numpy dtype use
`ndr_reader_tiffstack.numpy_dtype()`.

### Image readers

| MATLAB reader | Status in Python | Notes |
|---|---|---|
| `ndr.reader.tiffstack` | **Ported** | Native multipage-TIFF reader, built on `tifffile`. Handles single multipage files, directory epochs, anchor/marker files, and `frametimes.txt` sidecars. |
| `ndr.reader.prairieview` | **Ported** | Native legacy Prairie View reader. Groups channels from `Cycle`/`Ch` file-name markers, reads real per-frame timestamps from the config, and reports raster-scan metadata. Reads legacy `.pcf`, modern PVScan XML, and legacy MM-era XML configs. No NANSEN dependency. |
| `ndr.reader.imagestack` | Not ported | Thin wrapper over NANSEN's `nansen.stack.ImageStack` — see the dependency note below. |

### Other readers and formats

| MATLAB | Status in Python | Notes |
|---|---|---|
| `ndr.reader.vld` + `+ndr/+format/+vld/` | **Ported** | VH Lab LabView `.vld`/`.vlh` reader. Both storage layouts (chunked and multiplexed), big-endian, with optional `precision`/`Scale` scaling. |
| 64-bit CED `.smrx` via `+ndr/+format/+ced/+sonpipe/` | Not ported | MATLAB 7745c1b and follow-ups. `sonpipe` shells out to an external binary installed by `ndr.setup.sonpipe`; the Python reader handles `.smr` only, through `neo`. |
| `+ndr/+format/+ced/isSON64.m` | Not ported | Companion to the `.smrx` path. |
| `+ndr/+format/+intan/detectRHD2000FileMode.m`, `getRHD2000FileList.m` | Not ported | Multi-file / directory-mode Intan helpers. The Python reader raises `NotImplementedError` for directory-mode epochs. |
| `+ndr/+file/fileobj.m` | Intentionally not ported | Wraps MATLAB `fopen`/`fread` handle semantics; Python uses file objects and numpy dtypes directly. |
| `+ndr/+docs/build.m`, `+ndr/+setup/sonpipe.m`, `+ndr/+fun/python_detect.m` | Intentionally not ported | MATLAB-only tooling. |

### Reader-types resource divergence

`resource/ndr_reader_types.json` still differs between the two repositories.
MATLAB registers 14 readers; Python now registers 13. Python is missing only
`imagestack` (the NANSEN-backed reader). The two files also still disagree on
some type aliases: MATLAB maps `smrx`/`ced-smrx` (the sonpipe-backed 64-bit
CED path, which Python does not implement) and `WMHS`.
The Python entries additionally carry fully-qualified `classname` values
including the Python class name, where MATLAB carries only the package path.

## External Library Dependencies

NDR-matlab vendors four MATLAB libraries under `lib/` — `NPMK` (Blackrock),
`TDTSDK`, `abfload`, and `sigTOOL` (CED). Python does not vendor equivalents;
the same coverage comes from `neo` (Blackrock, CED) and `pyabf` (Axon).

NDR-matlab additionally declares two *runtime* requirements in
`tools/requirements.txt`, installed by `matbox.installRequirements` (including
on CI): `vhlab-toolbox-matlab` and
[NANSEN](https://github.com/VervaekeLab/NANSEN).

**NANSEN is required by exactly one reader: `ndr.reader.imagestack`.** That
class is a wrapper over the public `nansen.stack.open` / `nansen.stack.ImageStack`
API and raises a clear error when NANSEN is absent from the MATLAB path. Every
other reader is NANSEN-free by design — `ndr.reader.tiffstack` states in its
own header that only the *design* (method names, dimension model) is adapted
from `nansen.stack.ImageStack` and that "no NANSEN source code is used, so no
NANSEN dependency is introduced". `ndr.reader.prairieview` likewise only
references NANSEN's adapter behavior for comparison.

That is what made this port straightforward: the frame API and the `tiffstack`
and `prairieview` readers were ported with no NANSEN involvement at all, using
`tifffile` and the standard library. Only an `imagestack` equivalent would need
a backend that dispatches across image formats, and NANSEN itself is a large
MATLAB application framework (GUI, pipelines, session management) whose Python
analogue would be a different library rather than a translation. `imagestack.m`
also carries an explicit caution to confirm NANSEN's license before lifting any
of its source into NDR. Accordingly, `ndr.reader.imagestack` is left
unported.

## Test Summary

```
38 passed, 28 xfailed, 13 skipped, 2 failed (pre-existing spikegadgets format test issues)
```

Example data files are included in `src/ndr/example_data/` for Intan (.rhd), CED (.smr), Axon (.abf), SpikeGadgets (.rec), and Blackrock (.nev, .ns2).
