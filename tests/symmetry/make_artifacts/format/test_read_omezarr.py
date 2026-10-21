"""Generate symmetry artifacts for the OME-Zarr format layer (readArray).

Zarr is lossless, so if MATLAB and Python decode the same chunks to
different values, one of them is wrong. This writes a dual-pyramid OME-Zarr
store with zarr-python itself (zarr v2, Blosc+Zstd: the codec the lab's data
uses), then records the SHA-256 of what ndr.format.omezarr.readArray returns
for a fixed set of regions. MATLAB's readArtifacts/readOmeZarr reads the same
store with its hand-rolled Blosc/Zstd reader and must reproduce every digest.

The regions and the artifact layout (example.zarr, metadata.json,
checksums.json) are those of the MATLAB half, so either port can read the
other's. Digests are computed over the bytes in MATLAB's typecast(x(:)) order
(see conftest.matlab_sha256).

Mirrors tools/tests/+ndr/+symmetry/+makeArtifacts/+format/readOmeZarr.m
"""

import json
import shutil

import numpy as np
import pytest

from tests.symmetry.conftest import PYTHON_ARTIFACTS, matlab_sha256

ARTIFACT_DIR = PYTHON_ARTIFACTS / "format" / "readOmeZarr" / "testReadOmeZarrArtifacts"

SEED = 20260906

# (name, pyramid, level, region) -- the MATLAB half's regionSpecs. A region is
# [start, stop] per axis (c, z, y, x), 1-based and inclusive; None is the
# whole array. Chosen to cover a whole level, one chunk, a chunk straddle,
# and the x-axis edge chunk (10 voxels in chunks of 4).
REGIONS = [
    ("whole_mean_1", "mean", 1, None),
    ("whole_max_1", "max", 1, None),
    ("whole_mean_2", "mean", 2, None),
    ("whole_max_2", "max", 2, None),
    ("inside_one_chunk", "mean", 1, [[1, 1, 1, 1], [1, 2, 3, 3]]),
    ("straddle", "mean", 1, [[1, 3, 3, 3], [1, 6, 6, 6]]),
    ("reach_edge_x", "mean", 1, [[1, 1, 1, 7], [1, 8, 8, 10]]),
    ("inside_edge_x", "mean", 1, [[1, 1, 1, 9], [1, 4, 4, 10]]),
]


class TestReadOmeZarr:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("zarr")
        pytest.importorskip("numcodecs")

    def test_read_ome_zarr_artifacts(self):
        from ndr.format.omezarr.readArray import readArray
        from tests._omezarr_fixture import make_example_fixture

        if ARTIFACT_DIR.exists():
            shutil.rmtree(ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True)

        fixture_dir, gt = make_example_fixture(
            ARTIFACT_DIR, with_chunks=True, seed=SEED, codec="blosc-zstd"
        )

        # The reader must return the ground truth before its digests are
        # worth publishing: a digest of wrong data would only test agreement.
        np.testing.assert_array_equal(readArray(str(fixture_dir), "mean", 1), gt.level0)
        np.testing.assert_array_equal(readArray(str(fixture_dir), "mean", 2), gt.mean_level1)
        np.testing.assert_array_equal(readArray(str(fixture_dir), "max", 2), gt.max_level1)

        metadata = {
            "fixtureName": fixture_dir.name,
            "level0Shape": list(gt.shape0),
            "level0ChunkShape": list(gt.chunk_shape0),
            "dtype": gt.dtype,
            "seed": SEED,
        }
        (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

        regions = []
        for name, pyramid, level, region in REGIONS:
            if region is None:
                got = readArray(str(fixture_dir), pyramid, level)
            else:
                got = readArray(str(fixture_dir), pyramid, level, Region=np.array(region))
            regions.append(
                {
                    "name": name,
                    "pyramid": pyramid,
                    "level": level,
                    "region": region if region is not None else [],
                    "sha256": matlab_sha256(got),
                }
            )
        (ARTIFACT_DIR / "checksums.json").write_text(json.dumps({"regions": regions}, indent=2))
