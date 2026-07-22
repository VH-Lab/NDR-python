"""Regression test: SpikeGadgets readers raise on short reads.

Previously rec_data was allocated with np.empty and a read past EOF broke the
loop, returning uninitialized heap memory as voltage (with a length that no
longer matched the truncated timestamp array). A short read must now raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ndr.format.spikegadgets.read_rec_config import read_rec_config
from ndr.format.spikegadgets.read_rec_trodeChannels import read_rec_trodeChannels

EXAMPLE_REC = Path(__file__).parents[1] / "src" / "ndr" / "example_data" / "example.rec"


@pytest.mark.skipif(not EXAMPLE_REC.exists(), reason="example.rec not available")
def test_short_read_raises():
    result = read_rec_config(str(EXAMPLE_REC))
    cfg = result[0] if isinstance(result, tuple) else result
    num_channels = int(cfg["numChannels"] if isinstance(cfg, dict) else cfg.numChannels)
    header_size = int(cfg["headerSize"] if isinstance(cfg, dict) else cfg.headerSize)
    sampling_rate = int(cfg["samplingRate"] if isinstance(cfg, dict) else cfg.samplingRate)

    # Request far more samples than the file holds; must raise, not return
    # uninitialized memory.
    with pytest.raises(EOFError):
        read_rec_trodeChannels(
            str(EXAMPLE_REC),
            num_channels,
            [1],
            sampling_rate,
            header_size,
            1,
            10_000_000,
        )
