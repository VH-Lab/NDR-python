# NDR-python Port Status

Status of the MATLAB → Python port of [NDR-matlab](https://github.com/VH-Lab/NDR-matlab).

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
| `numpy` | All readers | Array operations |

## Test Summary

```
38 passed, 28 xfailed, 13 skipped, 2 failed (pre-existing spikegadgets format test issues)
```

Example data files are included in `src/ndr/example_data/` for Intan (.rhd), CED (.smr), Axon (.abf), SpikeGadgets (.rec), and Blackrock (.nev, .ns2).
