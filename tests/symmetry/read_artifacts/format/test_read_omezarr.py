"""Read and verify symmetry artifacts for the OME-Zarr format layer (readArray).

Parameterized over both artifact sources. Each source ships its own
example.zarr inside its artifact directory, plus the SHA-256 of what its own
readArray returned for each region. This reads that store with the Python
readArray and checks every digest: a mismatch means the two ports decode the
same lossless bytes differently, so one of them is wrong.

Skips when the artifact directory for a source does not exist; under
NDR_SYMMETRY_CHECK_STRICT (CI) that skip fails, see tests/symmetry/conftest.py.

Mirrors tools/tests/+ndr/+symmetry/+readArtifacts/+format/readOmeZarr.m
"""

import json

import numpy as np
import pytest

from tests.symmetry.conftest import SOURCE_TYPES, SYMMETRY_BASE, matlab_sha256


@pytest.fixture(params=SOURCE_TYPES)
def source_type(request):
    return request.param


def _artifact_dir(source_type):
    return SYMMETRY_BASE / source_type / "format" / "readOmeZarr" / "testReadOmeZarrArtifacts"


class TestReadOmeZarr:
    def test_checksums(self, source_type):
        pytest.importorskip("zarr")
        from ndr.format.omezarr.readArray import readArray

        artifact_dir = _artifact_dir(source_type)
        if not artifact_dir.exists():
            pytest.skip(f"No artifacts from {source_type}")

        metadata = json.loads((artifact_dir / "metadata.json").read_text())
        fixture_dir = artifact_dir / metadata["fixtureName"]
        assert fixture_dir.is_dir(), f"fixture {fixture_dir} missing"

        # The point is the Blosc+Zstd decode path; an uncompressed fixture
        # would compare nothing interesting.
        zarray = json.loads((fixture_dir / "0" / ".zarray").read_text())
        assert zarray["compressor"]["id"] == "blosc"
        assert zarray["compressor"]["cname"] == "zstd"

        entries = json.loads((artifact_dir / "checksums.json").read_text())["regions"]
        if isinstance(entries, dict):  # MATLAB jsonencode of a one-element array
            entries = [entries]
        assert entries, f"no regions in {source_type} checksums.json"

        mismatches = []
        for e in entries:
            region = e["region"]
            if region is None or len(region) == 0:
                got = readArray(str(fixture_dir), e["pyramid"], int(e["level"]))
            else:
                got = readArray(
                    str(fixture_dir), e["pyramid"], int(e["level"]), Region=np.array(region)
                )
            actual = matlab_sha256(got)
            if actual != e["sha256"].lower():
                mismatches.append(f'{e["name"]} ({e["pyramid"]} level {e["level"]})')

        assert not mismatches, (
            f"SHA-256 mismatch against {source_type} for: {', '.join(mismatches)}. "
            "The two ports decode the same lossless bytes differently, so one "
            "of the readers is wrong."
        )
