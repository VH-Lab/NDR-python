# NDR-python

Neuroscience Data Reader - Python implementation.

A faithful Python port of [NDR-matlab](https://github.com/VH-Lab/NDR-matlab), following a lead-follow architecture where MATLAB is the source of truth.

## Overview

NDR (Neuroscience Data Reader) is a lower-level data-reading library used by [NDI](https://github.com/VH-Lab/NDI-matlab). It provides:

- An abstract base reader class (`ndr.reader.base`)
- A high-level reader wrapper (`ndr.reader`)
- Format-specific reader subclasses (Intan RHD, Axon ABF, CED SMR, etc.)
- Format handler packages with low-level file I/O
- Time, string, data, and file utilities

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from ndr.reader_wrapper import Reader

# Create a reader for Intan RHD files
r = Reader("intan")

# Read channels from an epoch
epochfiles = ["/path/to/data.rhd"]
channels = r.getchannelsepoch(epochfiles)
data, time = r.read(epochfiles, "ai1-3", t0=0, t1=10)
```

## Supported Formats

| Format | Reader Class | Status |
|--------|-------------|--------|
| Intan RHD | `ndr.reader.intan_rhd.IntanRHD` | Implemented |
| Axon ABF | `ndr.reader.axon_abf.AxonABF` | Implemented (requires `pyabf`) |
| CED SMR | `ndr.reader.ced_smr.CedSMR` | Implemented (requires `neo`) |
| SpikeGadgets REC | `ndr.reader.spikegadgets_rec.SpikeGadgetsRec` | Stub |
| TDT SEV | `ndr.reader.tdt_sev.TdtSev` | Stub |
| Neo | `ndr.reader.neo.NeoReader` | Stub |
| White Matter | `ndr.reader.whitematter.WhiteMatter` | Stub |
| BJG | `ndr.reader.bjg.BJG` | Stub |
| Dabrowska | `ndr.reader.dabrowska.Dabrowska` | Stub |

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run symmetry tests (cross-language verification)
pytest tests/symmetry/ -v
```

## Architecture

See [AGENTS.md](AGENTS.md) and [docs/developer_notes/](docs/developer_notes/) for details on the lead-follow architecture, bridge YAML protocol, and porting guidelines.

## License

CC-BY-NC-SA-4.0
