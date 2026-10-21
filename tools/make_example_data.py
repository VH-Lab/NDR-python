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
    example_video.avi                  6-frame 12x16 uncompressed RGB AVI
    example_video_grey.avi             the same, 8-bit greyscale (grey palette)
"""

from __future__ import annotations

import struct
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

# --- video ----------------------------------------------------------------
VIDEO_FRAMES = 6
VIDEO_HEIGHT = 12
# A multiple of 4, so DIB rows need no padding in either bit depth.
VIDEO_WIDTH = 16
VIDEO_FPS = 10


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


def video_rgb() -> np.ndarray:
    """Deterministic RGB video, shape (T, Y, X, 3) uint8, distinct per frame/pixel/channel."""
    f, row, col = np.meshgrid(
        np.arange(VIDEO_FRAMES), np.arange(VIDEO_HEIGHT), np.arange(VIDEO_WIDTH), indexing="ij"
    )
    return np.stack([20 * f + col, 10 * row + f, 200 - 10 * f - col], axis=-1).astype(np.uint8)


def video_grey() -> np.ndarray:
    """Deterministic greyscale video, shape (T, Y, X) uint8."""
    f, row, col = np.meshgrid(
        np.arange(VIDEO_FRAMES), np.arange(VIDEO_HEIGHT), np.arange(VIDEO_WIDTH), indexing="ij"
    )
    return (30 * f + 4 * row + col).astype(np.uint8)


def _chunk(fourcc: bytes, payload: bytes) -> bytes:
    pad = b"\0" if len(payload) % 2 else b""
    return fourcc + struct.pack("<I", len(payload)) + payload + pad


def _list(kind: bytes, payload: bytes) -> bytes:
    return _chunk(b"LIST", kind + payload)


def write_avi(path: Path, frames: np.ndarray) -> None:
    """Write an uncompressed (BI_RGB) AVI, laid out as MATLAB's VideoWriter does.

    ``frames`` is (T, Y, X, 3) RGB for a 24-bit file, or (T, Y, X) for an
    8-bit file with a 256-entry grey palette ('Grayscale AVI'). Frames are
    stored bottom-up (positive biHeight), the canonical DIB orientation that
    every AVI reader handles, rather than FFmpeg's top-down negative height.
    """
    t, h, w = frames.shape[:3]
    grey = frames.ndim == 3
    bitcount = 8 if grey else 24
    framebytes = w * h * bitcount // 8

    avih = struct.pack(
        "<IIIIIIIIII16x",
        round(1e6 / VIDEO_FPS),  # dwMicroSecPerFrame
        framebytes * VIDEO_FPS,  # dwMaxBytesPerSec
        0,  # dwPaddingGranularity
        0x10,  # dwFlags: AVIF_HASINDEX
        t,  # dwTotalFrames
        0,  # dwInitialFrames
        1,  # dwStreams
        framebytes,  # dwSuggestedBufferSize
        w,
        h,
    )
    strh = (
        b"vids"
        + b"DIB "
        + struct.pack(
            "<IHHIIIIIIIIhhhh",
            0,
            0,
            0,
            0,
            1,
            VIDEO_FPS,
            0,
            t,
            framebytes,
            0xFFFFFFFF,
            0,
            0,
            0,
            w,
            h,
        )
    )
    strf = struct.pack("<IiiHHIIiiII", 40, w, h, 1, bitcount, 0, framebytes, 0, 0, 0, 0)
    if grey:
        strf = strf[:32] + struct.pack("<II", 256, 0)  # biClrUsed = 256
        strf += b"".join(bytes([k, k, k, 0]) for k in range(256))  # RGBQUAD: B,G,R,0
    hdrl = _list(
        b"hdrl",
        _chunk(b"avih", avih) + _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)),
    )

    movi_chunks = []
    idx1 = b""
    offset = 4  # idx1 offsets count from the 'movi' fourcc
    for k in range(t):
        img = frames[k][::-1]  # bottom-up rows
        data = (img if grey else img[..., ::-1]).tobytes()  # DIB colour order is B,G,R
        c = _chunk(b"00db", data)
        idx1 += b"00db" + struct.pack("<III", 0x10, offset, len(data))  # AVIIF_KEYFRAME
        offset += len(c)
        movi_chunks.append(c)
    body = b"AVI " + hdrl + _list(b"movi", b"".join(movi_chunks)) + _chunk(b"idx1", idx1)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_video(out: Path) -> None:
    """Write the RGB and greyscale example videos."""
    write_avi(out / "example_video.avi", video_rgb())
    write_avi(out / "example_video_grey.avi", video_grey())


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
        write_video(out)
        print(f"wrote example data into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
