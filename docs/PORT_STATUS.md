# NDR-python Port Status

Status of the MATLAB → Python port of [NDR-matlab](https://github.com/VH-Lab/NDR-matlab).

## Naming Convention

Python class names are a mechanical mapping of the MATLAB originals:

| MATLAB path | Python module | Python class |
|---|---|---|
| `+ndr/+reader/base.m` | `ndr.reader.base` | `Base` |
| `+ndr/+reader/intan_rhd.m` | `ndr.reader.intan_rhd` | `IntanRHD` |
| `+ndr/+reader/ced_smr.m` | `ndr.reader.ced_smr` | `CedSMR` |
| `+ndr/+reader/axon_abf.m` | `ndr.reader.axon_abf` | `AxonABF` |
| `+ndr/+reader/neo.m` | `ndr.reader.neo` | `NeoReader` |
| `+ndr/+reader/spikegadgets_rec.m` | `ndr.reader.spikegadgets_rec` | `SpikeGadgetsRec` |
| `+ndr/+reader/tdt_sev.m` | `ndr.reader.tdt_sev` | `TdtSev` |
| `+ndr/+reader/bjg.m` | `ndr.reader.bjg` | `BJG` |
| `+ndr/+reader/dabrowska.m` | `ndr.reader.dabrowska` | `Dabrowska` |
| `+ndr/+reader/whitematter.m` | `ndr.reader.whitematter` | `WhiteMatter` |

File names are preserved exactly from MATLAB (snake\_case). Class names are PascalCase conversions of the same.

## Reader Status

| Reader | getchannelsepoch | t0\_t1 | samplerate | readchannels\_epochsamples | readevents\_epochsamples\_native | read | Tests |
|---|---|---|---|---|---|---|---|
| **IntanRHD** | Yes | Yes | Yes | Yes (single-file) | Stub (empty) | Yes | 6 pass |
| **CedSMR** | Yes | Yes | Yes | Yes | Yes | Yes (via Base) | 14 pass |
| **AxonABF** | Yes | Yes | Yes | Yes | Stub (empty) | Yes (via Base) | 6 pass |
| **NeoReader** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | 24 xfail |
| **SpikeGadgetsRec** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **TdtSev** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **BJG** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **Dabrowska** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |
| **WhiteMatter** | Stub (empty) | Stub | Stub | NotImplementedError | Stub (empty) | No | skipped |

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

The top-level `ndr.Reader` class (`reader_wrapper.py`) wraps any format-specific reader and adds:

| Feature | Status |
|---|---|
| `read()` convenience method | Implemented |
| `readevents_epochsamples()` with derived events (dep, den, dimp, dimn) | Implemented |
| Delegation to underlying reader | Implemented |

## External Dependencies

| Dependency | Used by | Purpose |
|---|---|---|
| `neo` | CedSMR, NeoReader | Read CED SMR/SON and Blackrock files |
| `pyabf` | AxonABF | Read Axon Binary Format files |
| `numpy` | All readers | Array operations |

## Test Summary

```
36 passed, 24 xfailed, 19 skipped, 2 failed (pre-existing spikegadgets format test issues)
```

Example data files are included in `src/ndr/example_data/` for Intan (.rhd), CED (.smr), Axon (.abf), SpikeGadgets (.rec), and Blackrock (.nev, .ns2).
