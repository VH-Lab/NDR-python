# NDR-python Audit Remediation — Results (2026-06-12)

> **Context for a reviewer / next agent.** One of **9 coordinated PRs** in the 2026-06 NDI
> ecosystem audit; **none are merged.** This repo's PR: **VH-Lab/NDR-python#6**.
> What's done here is the fail-fast stub marking + alias reconciliation + LICENSE.
> **Two readers are deliberately left unimplemented (`implemented:false`):** the
> `spikegadgets_rec` binary sample reader (unresolved `.rec` byte-alignment — needs the
> format spec or a MATLAB cross-check; config layer works) and `neo` (needs `python-neo`
> + files; no MATLAB ref). Both are detailed below — that's the work to finish here.

Branch `audit/ndr-python-2026-06`, off `origin/main` (`896ed63`). VH-Lab
fork-and-PR (fork at review time).

## Findings addressed (audit §6.2)

| # | Severity | Status | Summary |
|---|----------|--------|---------|
| 6.2-5 (stub marking) | High | **Done** | The six not-yet-implemented readers (`spikegadgets_rec`, `whitematter`, `bjg`, `tdt_sev`, `dabrowska`, `neo`) raised `NotImplementedError` deep inside a read. They are now marked `"implemented": false` in `ndr_reader_types.json`, and the reader dispatch raises a clear `NotImplementedError` at construction (naming the reader) so callers fail fast. |
| 6.2-6 (alias registry) | Medium | **Done** | The Python alias registry diverged from NDR-matlab's. Reconciled each reader's alias list to the union of both languages' aliases (`RHD`/`intanRHD`/`son`/`ced-smr`/`SpikeGadgetsREC`/`WMHS`/`bjg_bin`/`dabrowska_mat`). The dispatch already matched case-insensitively. |
| 6.2-8 (LICENSE) | Medium | **Done** | Added `LICENSE` (MIT) matching the NDR-matlab counterpart. (The audit said "CC BY-NC-SA matching the MATLAB counterpart," but NDR-matlab is in fact MIT — matched the real counterpart.) |
| 6.2-5 (implement neo + spikegadgets_rec) | High | **Partial — flagged** | See below. |

## spikegadgets_rec — config/metadata validated; binary sample read FLAGGED

The `.rec` **format helpers** (`+ndr/+format/+spikegadgets/`) were already
ported. The config parser (`read_rec_config`) is validated correct against the
real `CS31_20170201_OdorPlace1short.rec` example: it returns `numChannels=120`,
`samplingRate=30000`, `headerSize=17`, 30 tetrodes — matching the file, including
the `saveDisplayedChanOnly` unused-channel remap path. So the reader's
config-derived methods (`getchannelsepoch`/`samplerate`/`t0_t1`) are sound.

The **binary sample reader** (`read_rec_trodeChannels`), however, has an
unresolved block-layout ambiguity that I could not pin down without the SpikeGadgets
`.rec` format spec or a MATLAB cross-check: the physical data block is 276 bytes
(the example file's data region is 60000 complete 276-byte blocks + a 39-byte
partial tail — i.e. 60000 samples = 2.0 s at 30 kHz, the block size is right and a
short partial final block is normal; it is NOT an exact multiple), but the MATLAB
reference's read layout implies `header(34) + timestamp(4)
+ channels(240) = 278` (`blockSizeBytes = header + 2 + channels` in the
reference disagrees with the +4-byte timestamp seek by 2 bytes). The per-sample
**timestamps come out implausible** on the real file (≈25 s span for 3000 samples;
trying `block_size-4` as the stride made it worse), which means the timestamp
offset — and possibly a ±2-byte channel-sample alignment — is wrong. The channel
values read in a plausible microvolt range, but plausible ≠ correctly aligned.

**Therefore `spikegadgets_rec` is left marked `implemented: false`** (fail-fast)
rather than shipped with a sample reader that may be subtly misaligned. Closing
it needs the `.rec` format spec or a MATLAB run to validate sample/timestamp
alignment bit-for-bit. The config/metadata layer is ready to wire once that is
resolved.

## neo — FLAGGED (not implemented)

`neo` is the gateway to blackrock/plexon/etc. via the `python-neo` package. It is
**not implemented** here: `python-neo` is a heavy dependency (not installed) and
there is no MATLAB reference (neo is Python-only), so a faithful generic
neo-rawio-backed reader would need multi-format real-data validation that is out
of reach in this environment. It is marked `implemented: false` so dispatch fails
fast. Recommended as a focused follow-up: implement on `neo.rawio`, validate
against an intan/axon/blackrock fixture.

## Validation

`PYTHONPATH=src python -m pytest tests/ --ignore=tests/symmetry` = **102 passed**
(incl. new `tests/test_reader_dispatch.py`, 13: stub fail-fast, implemented
readers construct, MATLAB aliases resolve, unknown → ValueError). black + ruff
clean.
