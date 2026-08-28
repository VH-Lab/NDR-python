#!/usr/bin/env python3
"""Generate the small example-data fixtures for the vld and image readers.

These fixtures back the cross-language symmetry tests, so the files must be
byte-identical in NDR-python and NDR-matlab. Everything here is deterministic
(no randomness, no timestamps), so re-running reproduces the same bytes.

Usage:
    python3 tools/make_example_data.py <target_example_data_dir> [...]

Each target directory receives:
    example.vld / example.vlh          VH Lab LabView, 3 ch, chunked layout
    example_movie.tif                  5-page 16x12 uint16 TIFF stack
    example_movie_frametimes.txt       one frame time per page, seconds
    prairieview/                       2 channels x 4 frames + legacy .pcf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

# --- vld ------------------------------------------------------------------
VLD_NCHANS = 3
VLD_SR = 1000.0
VLD_SAMPLES_PER_CHUNK = 100
VLD_NCHUNKS = 5
VLD_TOTAL = VLD_SAMPLES_PER_CHUNK * VLD_NCHUNKS

# --- tiffstack ------------------------------------------------------------
TIFF_PAGES = 5
TIFF_HEIGHT = 16
TIFF_WIDTH = 12
TIFF_FRAME_PERIOD = 0.05

# --- prairieview ----------------------------------------------------------
PV_CHANNELS = 2
PV_FRAMES = 4
PV_HEIGHT = 8
PV_WIDTH = 6
# Deliberately NOT frame_period / lines_per_frame: a real raster scan has
# flyback overhead between lines, so the config's explicit ScanLine period is
# shorter than the naive derivation. Keeping them distinct means the symmetry
# artifact can tell "read from the config" apart from "derived".
PV_LINE_PERIOD_US = 1500.0
PV_FRAME_PERIOD_US = 16000.0


def vld_signal() -> np.ndarray:
    """Deterministic per-channel waveform, shape (VLD_TOTAL, VLD_NCHANS)."""
    t = np.arange(VLD_TOTAL, dtype=np.float64) / VLD_SR
    return np.column_stack(
        [np.sin(2 * np.pi * (c + 1) * 10.0 * t) * (c + 1) for c in range(VLD_NCHANS)]
    )


def write_vld(out: Path) -> None:
    """Write the chunked-layout .vld plus its .vlh header."""
    data = vld_signal()

    # Chunked layout: SamplesPerChunk samples of ch1, then ch2, ... per chunk.
    with open(out / "example.vld", "wb") as f:
        for c in range(VLD_NCHUNKS):
            block = data[c * VLD_SAMPLES_PER_CHUNK : (c + 1) * VLD_SAMPLES_PER_CHUNK, :]
            for ch in range(VLD_NCHANS):
                f.write(block[:, ch].astype(">f8").tobytes())

    (out / "example.vlh").write_text(
        "ChannelString:\t/dev/ai0,/dev/ai1,/dev/ai2\n"
        f"NumChans:\t{VLD_NCHANS}\n"
        f"SamplingRate:\t{int(VLD_SR)}\n"
        f"SamplesPerChunk:\t{VLD_SAMPLES_PER_CHUNK}\n"
        "Multiplexed:\t0\n"
    )


def tiff_stack() -> np.ndarray:
    """Deterministic 5-page uint16 stack, distinct in every page and pixel."""
    page, row, col = np.meshgrid(
        np.arange(TIFF_PAGES), np.arange(TIFF_HEIGHT), np.arange(TIFF_WIDTH), indexing="ij"
    )
    return (page * 1000 + row * 10 + col).astype(np.uint16)


def write_tiffstack(out: Path) -> None:
    """Write the multipage TIFF and its frame-times sidecar."""
    # Uncompressed baseline TIFF, so MATLAB's imread reads it without extras.
    tifffile.imwrite(out / "example_movie.tif", tiff_stack(), compression=None)
    (out / "example_movie_frametimes.txt").write_text(
        "".join(f"{i * TIFF_FRAME_PERIOD:.6f}\n" for i in range(TIFF_PAGES))
    )


def write_prairieview(out: Path) -> None:
    """Write a Prairie View epoch: per-channel/frame TIFFs plus a legacy .pcf."""
    pv = out / "prairieview"
    pv.mkdir(exist_ok=True)

    for frame in range(1, PV_FRAMES + 1):
        for chan in range(1, PV_CHANNELS + 1):
            row, col = np.meshgrid(np.arange(PV_HEIGHT), np.arange(PV_WIDTH), indexing="ij")
            img = (chan * 1000 + frame * 100 + row * 10 + col).astype(np.uint16)
            tifffile.imwrite(
                pv / f"example_Cycle001_Ch{chan}_{frame:06d}.tif", img, compression=None
            )

    timestamps = "".join(f"img{i + 1}={i * PV_FRAME_PERIOD_US:.1f}\n" for i in range(PV_FRAMES))
    (pv / "example_Main.pcf").write_text(
        "[Main]\n"
        f"Lines per frame={PV_HEIGHT}\n"
        f"Pixels per line={PV_WIDTH}\n"
        f"Frame period (us)={PV_FRAME_PERIOD_US:.1f}\n"
        f"ScanLine period (us)={PV_LINE_PERIOD_US:.1f}\n"
        f"Total images={PV_FRAMES}\n"
        "\n"
        "[Image TimeStamp (us)]\n" + timestamps
    )


def main(targets: list[str]) -> int:
    if not targets:
        print(__doc__)
        return 1
    for target in targets:
        out = Path(target)
        out.mkdir(parents=True, exist_ok=True)
        write_vld(out)
        write_tiffstack(out)
        write_prairieview(out)
        print(f"wrote example data into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
